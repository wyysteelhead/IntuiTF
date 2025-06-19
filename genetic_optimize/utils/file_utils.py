
import json
import os
import random

def load_config(config_path):
    """
    从本地 JSON 文件中加载配置，包括 API 密钥。
    """
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    return config


def load_prompt(prompt_path):
    """
    从外部 TXT 文件中加载提示文本。
    """
    assert os.path.isfile(prompt_path)
    with open(prompt_path, "r", encoding="utf-8") as prompt_file:
        return prompt_file.read().strip()


def load_aspects(aspect_path, quality_metrics, text_metrics):
    with open(aspect_path, "r") as file:
        file_ = json.load(file)
    aspects = file_['ASPECTS']
    modes = file_['MODES']
    if quality_metrics is not None:
        modes['quality'] = quality_metrics
    if text_metrics is not None:
        modes['text'] = text_metrics
    # Generate the required formatted string for each aspect
    # formatted_aspects = [f"{aspect['index']}. {aspect['name']}: {aspect['description']}" for aspect in aspects]

    return aspects, modes

#deprecated

def load_prompt_with_instruct(instruct_path, instruct_number, prompt_path):
    text_prompt = load_instruct(instruct_path, instruct_number)
    prompt = load_prompt(prompt_path)
    postfix_temp = 'Following is the text prompt from which these two volume rendered image are described:\n"{}"\nPlease compare these two image as instructed.'
    postfix = postfix_temp.format(text_prompt)
    return prompt + "\n" + postfix


def load_instruct(instruct_path, instruct_number=None):
    """
    从 JSON 文件中加载指令。
    
    参数:
        instruct_path (str): JSON 文件路径。
        instruct_number (int, optional): 指定加载的指令编号。如果为 None，则随机选择一条指令。
    
    返回:
        dict: 包含指令名称和内容的字典。例如：{"name": "instruct1", "content": "..."}
    """
    # 加载 JSON 文件
    with open(instruct_path, 'r') as f:
        data = json.load(f)
    
    # 获取指令列表
    instructions = data.get("instructions", [])
    
    if not instructions:
        raise ValueError("No instructions found in the JSON file.")
    
    if instruct_number is None:
        return None
    
    # 如果 instruct_number 是数字字符串，转换为整数并作为索引使用
    if instruct_number.isdigit():
        index = int(instruct_number)
        if index < 0 or index >= len(instructions):
            raise ValueError(f"Invalid instruct_number: {index}. Must be between 0 and {len(instructions) - 1}.")
        return instructions[index].get("content")
    
    # 如果 instruct_number 是非数字字符串，根据 name 检索指令
    else:
        for instr in instructions:
            if instr.get("name") == instruct_number:
                return instr.get("content")
        raise ValueError(f"Instruction with name '{instruct_number}' not found.")
    