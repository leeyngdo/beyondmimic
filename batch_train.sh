#!/bin/bash

TARGET_DIR="/home/jovyan/whole_body_tracking/LAFAN1_Retargeting_Dataset/g1"
GPUS=(0 1 2 3)
JOBS_PER_GPU=2
MAX_JOBS=$((${#GPUS[@]} * JOBS_PER_GPU))  # 8

i=0

for file in "$TARGET_DIR"/*.csv; do
    motion_name=$(basename "$file" .csv)

    gpu_index=$((i % MAX_JOBS))
    gpu_id=${GPUS[$((gpu_index % ${#GPUS[@]}))]}

    echo "Running $motion_name on GPU $gpu_id..."

    CUDA_VISIBLE_DEVICES=$gpu_id python scripts/rsl_rl/train.py \
        --task=Tracking-Flat-G1-v0 \
        --registry_name wandb-registry-motion/lafan1_${motion_name} \
        --headless \
        --logger wandb \
        --log_project_name g1_expert \
        --run_name lafan1_${motion_name} &

    ((i++))

    # 8개씩 실행 후 대기
    if (( i % MAX_JOBS == 0 )); then
        wait
    fi
done

wait
echo "All training jobs completed."