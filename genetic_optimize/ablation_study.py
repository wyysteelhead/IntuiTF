#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ast
import os
import sys
sys.path.append('./DiffDVR/pytests')
sys.path.append('./DiffDVR')
sys.path.append('.')
import pickle
import json
import argparse
from typing import List, Dict, Any
import glob
from genetic_optimize.states.bound import Bound
from genetic import GeneticAlgorithm
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="合并多个 .pkl 文件中的优质个体到一个新种群并进行ELO评分")
    parser.add_argument('--input_dirs', type=str, nargs='+', required=True, 
                        help="包含 .pkl 文件的目录列表")
    parser.add_argument('--output_file', type=str, required=True,
                        help="输出合并后的种群文件路径")
    parser.add_argument('--config_file', type=str, required=True,
                        help="用于加载体绘制参数的配置文件")
    parser.add_argument('--top_n', type=int, default=10,
                        help="从每个种群中选取的顶级个体数量")
    parser.add_argument('--result_json', type=str, required=True,
                        help="输出各种群评估结果的JSON文件路径")
    parser.add_argument('--base_url', type=str, default="", 
                        help="ELO评估所需的API基础URL")
    parser.add_argument('--api_key', type=str, default="", 
                        help="ELO评估所需的API密钥")
    parser.add_argument('--prompt_folder', type=str, default="", 
                        help="ELO评估所需的提示文件夹")
    parser.add_argument('--bg_color', type=str, default="(255, 255, 255)", help="Background color for the image.")
    
    return parser.parse_args()

def load_population_from_pkl(pkl_path: str) -> tuple:
    """加载一个 .pkl 文件中的种群数据"""
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        population = data.get('population', [])
        iteration = data.get('iteration', 0)
        mode = data.get('mode', "quality")
        return population, iteration, mode
    except Exception as e:
        print(f"加载 {pkl_path} 失败: {str(e)}")
        return [], 0, "quality"

def sort_population_by_rating(population):
    """按 rating 从高到低排序种群"""
    return sorted(population, key=lambda x: -x.rating)

def main():
    args = parse_args()
    args.bg_color = ast.literal_eval(args.bg_color)
    
    # 存储所有选出的优质个体和它们的来源
    elite_individuals = []
    source_map = {}
    
    # 存储每个来源的个体ID
    source_individuals = {}
    
    print(f"将从每个种群中选择 {args.top_n} 个评分最高的个体")
    
    # 加载体配置
    bound = Bound(args.config_file)
    # 遍历每个输入目录
    for input_dir in args.input_dirs:
        # 查找目录中所有的 .pkl 文件
        pkl_files = glob.glob(os.path.join(input_dir, "*.pkl"))
        
        print(f"在 {input_dir} 中找到 {len(pkl_files)} 个 .pkl 文件")
        
        for pkl_file in pkl_files:
            # 生成来源标识符
            source_id = f"{os.path.basename(input_dir)}/{os.path.basename(pkl_file)}"
            print(f"处理 {source_id}...")
            
            # 加载种群
            population, iteration, mode = load_population_from_pkl(pkl_file)
            
            if not population:
                print(f"跳过空种群: {source_id}")
                continue
            
            # 确保种群按 rating 排序
            sorted_population = sort_population_by_rating(population)
            
            # 选取前 N 个个体
            top_n = min(args.top_n, len(sorted_population))
            top_individuals = sorted_population[:top_n]
            
            # 初始化用于存储该来源个体的列表
            source_individuals[source_id] = []
            
            # 为每个个体添加源标识符并添加到精英列表
            for i, ind in enumerate(top_individuals):
                ind.source_id = source_id  # 添加源标识符
                ind.original_rank = i
                ind.original_rating = ind.rating
                
                # 添加到精英个体列表
                elite_individuals.append(ind)
                source_individuals[source_id].append(ind)
            
            print(f"从 {source_id} 中选取了 {top_n} 个个体")
    
    print(f"总共选取了 {len(elite_individuals)} 个精英个体")
    
    # 如果没有找到个体，退出
    if not elite_individuals:
        print("未找到任何个体，退出")
        return
    
    # 对所有精英个体重排序
    final_elite = sort_population_by_rating(elite_individuals)
    
    # 初始化遗传算法实例，用于ELO评估
    ga = GeneticAlgorithm(
        bound=bound,
        base_url=args.base_url,
        api_key=args.api_key,
        prompt_folder=args.prompt_folder,
        save_path=os.path.dirname(args.output_file)
    )
    
    # 重设所有个体的ID (在ELO评估前先重置ID)
    for i, ind in enumerate(final_elite):
        ind.reset_matching(i)  # 重置匹配信息和设置新ID
        ind.load_render_settings(bound=bound, volume=ga.volume, gradient=ga.gradient, step_size=ga.setting.get_stepsize(), bg_color = args.bg_color)

    # 对合并后的种群进行ELO评分
    print("对合并后的种群进行ELO评分...")
    sorted_population = ga.par_elo_tournament(population=final_elite, mode="quality", save_path=ga.save_path, num_workers=None)
    final_elite = sorted_population
    
    # 计算每个来源种群的平均评分和排名统计
    source_ratings = {}

    # 首先获取最终排名信息 - 从整体排名角度
    final_rankings = {}
    for rank, ind in enumerate(final_elite):
        if hasattr(ind, 'source_id'):
            final_rankings[ind.id] = rank + 1  # 排名从1开始

    # 然后处理每个来源
    for source_id in source_individuals.keys():
        # 找出所有属于该来源的个体
        source_inds = [ind for ind in final_elite if hasattr(ind, 'source_id') and ind.source_id == source_id]
        
        if source_inds:
            # 计算评分统计信息
            avg_rating = sum(ind.rating for ind in source_inds) / len(source_inds)
            max_rating = max(ind.rating for ind in source_inds)
            min_rating = min(ind.rating for ind in source_inds)
            
            # 计算排名统计信息
            rankings = [final_rankings[ind.id] for ind in source_inds]
            avg_ranking = sum(rankings) / len(rankings)
            best_ranking = min(rankings)  # 最低数值代表最高排名
            worst_ranking = max(rankings)  # 最高数值代表最低排名
            
            source_ratings[source_id] = {
                "average_rating": avg_rating,
                "max_rating": max_rating, 
                "min_rating": min_rating,
                "num_individuals": len(source_inds),
                "individual_ratings": [ind.rating for ind in source_inds],
                # 新增排名统计
                "average_ranking": avg_ranking,
                "best_ranking": best_ranking,    # 最好的排名（数值最小）
                "worst_ranking": worst_ranking,  # 最差的排名（数值最大）
                "individual_rankings": rankings
            }
            print(f"来源 {source_id}:")
            print(f"  平均评分: {avg_rating:.2f}, 个体数: {len(source_inds)}")
            print(f"  排名统计: 平均排名: {avg_ranking:.1f}, 最佳排名: {best_ranking}, 最低排名: {worst_ranking}")

    # 按平均评分对来源进行排序
    sorted_sources = sorted(source_ratings.items(), key=lambda x: -x[1]["average_rating"])

    # 构造结果输出
    results = {
        "sources": source_ratings,
        "ranking": [
            {
                "source": source_id, 
                "avg_rating": data["average_rating"],
                "avg_ranking": data["average_ranking"],
                "best_ranking": data["best_ranking"],
                "worst_ranking": data["worst_ranking"]
            } 
            for source_id, data in sorted_sources
        ]
    }

if __name__ == "__main__":
    main()