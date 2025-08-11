
import json
import os
import random

def load_config(config_path):
    """
    Load configuration from local JSON file, including API keys.
    """
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    return config


def load_prompt(prompt_path):
    """
    Load prompt text from external TXT file.
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
    Load instructions from JSON file.
    
    Parameters:
        instruct_path (str): JSON file path.
        instruct_number (int, optional): Specified instruction number to load. If None, randomly select an instruction.
    
    Returns:
        dict: Dictionary containing instruction name and content. Example: {"name": "instruct1", "content": "..."}
    """
    # Load JSON file
    with open(instruct_path, 'r') as f:
        data = json.load(f)
    
    # Get instruction list
    instructions = data.get("instructions", [])
    
    if not instructions:
        raise ValueError("No instructions found in the JSON file.")
    
    if instruct_number is None:
        return None
    
    # If instruct_number is a numeric string, convert to integer and use as index
    if instruct_number.isdigit():
        index = int(instruct_number)
        if index < 0 or index >= len(instructions):
            raise ValueError(f"Invalid instruct_number: {index}. Must be between 0 and {len(instructions) - 1}.")
        return instructions[index].get("content")
    
    # If instruct_number is a non-numeric string, retrieve instruction by name
    else:
        for instr in instructions:
            if instr.get("name") == instruct_number:
                return instr.get("content")
        raise ValueError(f"Instruction with name '{instruct_number}' not found.")
    
def get_project_root():
    """Return project root directory (assuming this code is in `project_root/src/` or similar subdirectory)"""
    current_dir = os.path.dirname(os.path.abspath(__file__))  # Absolute path of current script
    # Search upward until finding project root directory (e.g., directory containing `.git`, `requirements.txt` or `volumes/`)
    while True:
        if (
            os.path.exists(os.path.join(current_dir, "volumes"))  # Check if volumes directory exists
            or os.path.exists(os.path.join(current_dir, ".git"))  # Or check Git root directory
            or os.path.exists(os.path.join(current_dir, "README.md"))  # Or other project root markers
        ):
            return current_dir
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:  # Already reached filesystem root directory
            raise FileNotFoundError("Could not find project root directory!")
        current_dir = parent_dir