#!/bin/bash

# Initialize conda for bash script
eval "$(conda shell.bash hook)"
conda activate genetic

# Function to run the command with specified parameters in a new tmux session
run_command_in_tmux() {
    local session_name=$1
    local config_file=$2
    local instruct_number=$3
    local save_path=$4
    local quality_metrics=$5
    local text_metrics=$6
    local text_interval=$7
    local bg_color=$8
    local population=$9
    local generations=$10
    local gpu_id=$11
    
    # 创建会话前先删除会话
    tmux kill-session -t "$session_name" 2>/dev/null
    # Create a new tmux session
    tmux new-session -d -s "$session_name"
    
    # Send the commands to the tmux session
    tmux send-keys -t "$session_name" "conda activate genetic" C-m
    tmux send-keys -t "$session_name" "echo Running task $session_name on GPU $gpu_id" C-m
    tmux send-keys -t "$session_name" "CUDA_VISIBLE_DEVICES=$gpu_id python genetic.py \
        --base_url [BASE_URL] \
        --api_key [API_KEY] \
        --config_file \"$config_file\" \
        --prompt_folder ./prompt \
        --population_size $population \
        --generations $generations \
        --save_interval 1 \
        --instruct_number \"$instruct_number\" \
        --save_path \"$save_path\" \
        --quality_metrics \"$quality_metrics\" \
        --text_metrics \"$text_metrics\" \
        --text_interval \"$text_interval\" \
        --bg_color \"$bg_color\" \
        --model_name \"[MODEL_NAME]\"" C-m # model_name is the model you want to use. Our code has been tested on Gemini and OpenAI API.
    
    echo "Started task in tmux session: $session_name on GPU $gpu_id"
}

run_command_in_tmux "feet7" \
    "./diffdvr/config-files/feet.json" \
    "feet7" \
    "./results/0612/gemini2/feet7" \
    "16,11,14" \
    "5" \
    "16" \
    "(255,255,255)" \
    "50" \
    "30" \
    "1"
