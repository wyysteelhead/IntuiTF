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
    parser = argparse.ArgumentParser(description="Merge high-quality individuals from multiple .pkl files into a new population and perform ELO scoring")
    parser.add_argument('--input_dirs', type=str, nargs='+', required=True, 
                        help="List of directories containing .pkl files")
    parser.add_argument('--output_file', type=str, required=True,
                        help="Output path for merged population file")
    parser.add_argument('--config_file', type=str, required=True,
                        help="Configuration file for loading volume rendering parameters")
    parser.add_argument('--top_n', type=int, default=10,
                        help="Number of top individuals to select from each population")
    parser.add_argument('--result_json', type=str, required=True,
                        help="JSON file path for outputting population evaluation results")
    parser.add_argument('--base_url', type=str, default="", 
                        help="Base API URL required for ELO evaluation")
    parser.add_argument('--api_key', type=str, default="", 
                        help="API key required for ELO evaluation")
    parser.add_argument('--prompt_folder', type=str, default="", 
                        help="Prompt folder required for ELO evaluation")
    parser.add_argument('--bg_color', type=str, default="(255, 255, 255)", help="Background color for the image.")
    
    return parser.parse_args()

def load_population_from_pkl(pkl_path: str) -> tuple:
    """Load population data from a .pkl file"""
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        population = data.get('population', [])
        iteration = data.get('iteration', 0)
        mode = data.get('mode', "quality")
        return population, iteration, mode
    except Exception as e:
        print(f"Failed to load {pkl_path}: {str(e)}")
        return [], 0, "quality"

def sort_population_by_rating(population):
    """Sort population by rating from high to low"""
    return sorted(population, key=lambda x: -x.rating)

def main():
    args = parse_args()
    args.bg_color = ast.literal_eval(args.bg_color)
    
    # Store all selected elite individuals and their sources
    elite_individuals = []
    source_map = {}
    
    # Store individual IDs for each source
    source_individuals = {}
    
    print(f"Will select {args.top_n} highest-rated individuals from each population")
    
    # Load volume configuration
    bound = Bound(args.config_file)
    # Iterate through each input directory
    for input_dir in args.input_dirs:
        # Find all .pkl files in the directory
        pkl_files = glob.glob(os.path.join(input_dir, "*.pkl"))
        
        print(f"Found {len(pkl_files)} .pkl files in {input_dir}")
        
        for pkl_file in pkl_files:
            # Generate source identifier
            source_id = f"{os.path.basename(input_dir)}/{os.path.basename(pkl_file)}"
            print(f"Processing {source_id}...")
            
            # Load population
            population, iteration, mode = load_population_from_pkl(pkl_file)
            
            if not population:
                print(f"Skipping empty population: {source_id}")
                continue
            
            # Ensure population is sorted by rating
            sorted_population = sort_population_by_rating(population)
            
            # Select top N individuals
            top_n = min(args.top_n, len(sorted_population))
            top_individuals = sorted_population[:top_n]
            
            # Initialize list to store individuals from this source
            source_individuals[source_id] = []
            
            # Add source identifier to each individual and add to elite list
            for i, ind in enumerate(top_individuals):
                ind.source_id = source_id  # Add source identifier
                ind.original_rank = i
                ind.original_rating = ind.rating
                
                # Add to elite individuals list
                elite_individuals.append(ind)
                source_individuals[source_id].append(ind)
            
            print(f"Selected {top_n} individuals from {source_id}")
    
    print(f"Selected a total of {len(elite_individuals)} elite individuals")
    
    # If no individuals found, exit
    if not elite_individuals:
        print("No individuals found, exiting")
        return
    
    # Re-sort all elite individuals
    final_elite = sort_population_by_rating(elite_individuals)
    
    # Initialize genetic algorithm instance for ELO evaluation
    ga = GeneticAlgorithm(
        bound=bound,
        base_url=args.base_url,
        api_key=args.api_key,
        prompt_folder=args.prompt_folder,
        save_path=os.path.dirname(args.output_file)
    )
    
    # Reset all individual IDs (reset IDs before ELO evaluation)
    for i, ind in enumerate(final_elite):
        ind.reset_matching(i)  # Reset matching information and set new ID
        ind.load_render_settings(bound=bound, volume=ga.volume, gradient=ga.gradient, step_size=ga.setting.get_stepsize(), bg_color = args.bg_color)

    # Perform ELO scoring on the merged population
    print("Performing ELO scoring on the merged population...")
    sorted_population = ga.par_elo_tournament(population=final_elite, mode="quality", save_path=ga.save_path, num_workers=None)
    final_elite = sorted_population
    
    # Calculate average ratings and ranking statistics for each source population
    source_ratings = {}

    # First get final ranking information - from overall ranking perspective
    final_rankings = {}
    for rank, ind in enumerate(final_elite):
        if hasattr(ind, 'source_id'):
            final_rankings[ind.id] = rank + 1  # Rankings start from 1

    # Then process each source
    for source_id in source_individuals.keys():
        # Find all individuals belonging to this source
        source_inds = [ind for ind in final_elite if hasattr(ind, 'source_id') and ind.source_id == source_id]
        
        if source_inds:
            # Calculate rating statistics
            avg_rating = sum(ind.rating for ind in source_inds) / len(source_inds)
            max_rating = max(ind.rating for ind in source_inds)
            min_rating = min(ind.rating for ind in source_inds)
            
            # Calculate ranking statistics
            rankings = [final_rankings[ind.id] for ind in source_inds]
            avg_ranking = sum(rankings) / len(rankings)
            best_ranking = min(rankings)  # Lowest number represents highest rank
            worst_ranking = max(rankings)  # Highest number represents lowest rank
            
            source_ratings[source_id] = {
                "average_rating": avg_rating,
                "max_rating": max_rating, 
                "min_rating": min_rating,
                "num_individuals": len(source_inds),
                "individual_ratings": [ind.rating for ind in source_inds],
                # Added ranking statistics
                "average_ranking": avg_ranking,
                "best_ranking": best_ranking,    # Best ranking (smallest value)
                "worst_ranking": worst_ranking,  # Worst ranking (largest value)
                "individual_rankings": rankings
            }
            print(f"Source {source_id}:")
            print(f"  Average rating: {avg_rating:.2f}, Number of individuals: {len(source_inds)}")
            print(f"  Ranking stats: Average rank: {avg_ranking:.1f}, Best rank: {best_ranking}, Worst rank: {worst_ranking}")

    # Sort sources by average rating
    sorted_sources = sorted(source_ratings.items(), key=lambda x: -x[1]["average_rating"])

    # Construct result output
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