#!/bin/bash

LOG_DIR="/home/jovyan/whole_body_tracking/logs/rsl_rl/g1_flat"
GPUS=(0 1 2 3)
JOBS_PER_GPU=2
MAX_JOBS=$((${#GPUS[@]} * JOBS_PER_GPU))  # 8

# All LAFAN1 motions
ALL_MOTIONS=(
    dance1_subject1
    dance1_subject2
    dance1_subject3
    dance2_subject1
    dance2_subject2
    dance2_subject3
    dance2_subject4
    dance2_subject5
    fallAndGetUp1_subject1
    fallAndGetUp1_subject4
    fallAndGetUp1_subject5
    fallAndGetUp2_subject2
    fallAndGetUp2_subject3
    fallAndGetUp3_subject1
    fight1_subject2
    fight1_subject3
    fight1_subject5
    fightAndSports1_subject1
    fightAndSports1_subject4
    jumps1_subject1
    jumps1_subject2
    jumps1_subject5
    run1_subject2
    run1_subject5
    run2_subject1
    run2_subject4
    sprint1_subject2
    sprint1_subject4
    walk1_subject1
    walk1_subject2
    walk1_subject5
    walk2_subject1
    walk2_subject3
    walk2_subject4
    walk3_subject1
    walk3_subject2
    walk3_subject3
    walk3_subject4
    walk3_subject5
    walk4_subject1
)

# Find incomplete motions (no log folder)
TODO=()
for m in "${ALL_MOTIONS[@]}"; do
    found=false
    for d in "$LOG_DIR"/*_lafan1_"$m"; do
        if [ -d "$d" ]; then
            found=true
            break
        fi
    done
    if ! $found; then
        TODO+=("$m")
    fi
done

echo "=== Remaining: ${#TODO[@]} motions ==="
for m in "${TODO[@]}"; do
    echo "  $m"
done
echo ""

i=0
for motion_name in "${TODO[@]}"; do
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

    if (( i % MAX_JOBS == 0 )); then
        wait
    fi
done

wait
echo "All remaining training jobs completed."
