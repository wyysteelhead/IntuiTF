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
    local style_image=$11
    local gpu_id=$12
    
    # 创建会话前先删除会话
    tmux kill-session -t "$session_name" 2>/dev/null
    # Create a new tmux session
    tmux new-session -d -s "$session_name"
    
    # Send the commands to the tmux session
    tmux send-keys -t "$session_name" "conda activate genetic" C-m
    tmux send-keys -t "$session_name" "echo Running task $session_name on GPU $gpu_id" C-m
    tmux send-keys -t "$session_name" "CUDA_VISIBLE_DEVICES=$gpu_id python genetic_optimize/genetic.py \
        --base_url [URL] \
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
        --style_image \"$style_image\" \
        --model_name \"[MODEL_NAME]\"" C-m
    
    echo "Started task in tmux session: $session_name on GPU $gpu_id" # model_name is the model you want to use. Our code has been tested on Gemini and OpenAI API.
}

# 运行多个任务，每个都在独立的tmux会话中
# 第一个参数是会话名称，最后一个参数是GPU ID

# GPU 0

run_command_in_tmux "engine1" \
    "/root/autodl-tmp/GeneticDVR/diffdvr/config-files/engine.json" \
    "engine1" \
    "/folderFromHost/results/0331/image/engine1" \
    "16,11,14" \
    "5" \
    "16" \
    "(255,255,255)" \
    "50" \
    "30" \
    "/folderFromHost/results/style_images/engine.png" \
    "1"

# run_command_in_tmux "jet1_image" \
#     "/root/autodl-tmp/GeneticDVR/diffdvr/config-files/jet.json" \
#     "jet1" \
#     "/folderFromHost/results/0329/image/jet1" \
#     "16,11,14" \
#     "5" \
#     "16" \
#     "(255,255,255)" \
#     "50" \
#     "30" \
#     "/root/autodl-tmp/GeneticDVR/prompt/star.png" \
#     "1"

# run_command_in_tmux "head_image" \
#     "/root/autodl-tmp/GeneticDVR/diffdvr/config-files/head.json" \
#     "head10" \
#     "/folderFromHost/results/0329/image/head10" \
#     "16,11,14" \
#     "5" \
#     "16" \
#     "(255,255,255)" \
#     "50" \
#     "30" \
#     "/root/autodl-tmp/GeneticDVR/prompt/head.png" \
#     "1"

# run_command_in_tmux "hurricane_image" \
#     "/root/autodl-tmp/GeneticDVR/diffdvr/config-files/hurricane.json" \
#     "head10" \
#     "/folderFromHost/results/0329/image/hurricane" \
#     "16,11,14" \
#     "5" \
#     "16" \
#     "(255,255,255)" \
#     "50" \
#     "30" \
#     "/root/autodl-tmp/GeneticDVR/prompt/star.png" \
#     "1"

# 你可以添加更多的任务，记得更改会话名称和GPU ID

# # 创建一个主会话用于显示所有子会话
# tmux new-session -d -s "main_view"

# # 计算布局 - 尝试创建一个近似方形的布局
# SESSION_COUNT=${#SESSION_NAMES[@]}
# COLS=$(echo "sqrt($SESSION_COUNT)" | bc)
# COLS=${COLS%.*}  # 去掉小数部分
# if [ $COLS -eq 0 ]; then COLS=1; fi
# ROWS=$(( (SESSION_COUNT + COLS - 1) / COLS ))  # 向上取整

# # 在主会话中创建足够的窗格
# CURRENT_PANE=0
# for (( i=0; i<$SESSION_COUNT; i++ )); do
#     if [ $i -gt 0 ]; then
#         # 确定是水平分割还是垂直分割
#         if [ $(( i % ROWS )) -eq 0 ]; then
#             tmux split-window -h -t main_view
#         else
#             tmux split-window -v -t main_view
#         fi
#     fi
#     # 在当前窗格中连接到相应的子会话
#     tmux send-keys -t main_view "tmux attach-session -t ${SESSION_NAMES[$i]}" C-m
#     CURRENT_PANE=$((CURRENT_PANE + 1))
# done

# # 选择平铺布局
# tmux select-layout -t main_view tiled

# echo "All tasks have been started in tmux sessions"
# echo "Individual sessions: use 'tmux attach -t SESSION_NAME' to view a specific session"
# echo "Combined view: use 'tmux attach -t main_view' to view all sessions in one screen"
# echo "Use 'tmux list-sessions' to see all running sessions"

# # 自动连接到主视图
# tmux attach -t main_view