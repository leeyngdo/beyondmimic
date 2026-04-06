#!/bin/bash
# Train one BeyondMimic expert per cluster using merged multi-motion data.
#
# Maintains MAX_JOBS concurrent jobs. When one finishes, the next starts
# on the least-loaded GPU.
#
# Usage:
#   bash batch_train_clusters.sh [cluster_mapping_dir]
#   bash batch_train_clusters.sh amass_g1/cluster_mapping

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MAPPING_DIR="${1:-amass_g1/cluster_mapping}"
REPS_FILE="$MAPPING_DIR/cluster_representatives.json"
POLL_INTERVAL=10800  # 3 hours

GPUS=(0 1 2 3)
JOBS_PER_GPU=2
MAX_JOBS=$((${#GPUS[@]} * JOBS_PER_GPU))  # 8

# Parse cluster IDs
mapfile -t ENTRIES < <(python -c "
import json
with open('$REPS_FILE') as f:
    reps = json.load(f)
for cid in sorted(reps, key=int):
    name = reps[cid].get('name', f'cluster_{cid}')
    print(f'{cid}|{name}')
")

echo "Training ${#ENTRIES[@]} cluster experts on GPUs ${GPUS[*]} (${JOBS_PER_GPU}/GPU = ${MAX_JOBS} parallel)"
echo "Poll interval: ${POLL_INTERVAL}s"

# Find the GPU with fewest running training jobs
find_least_loaded_gpu() {
    local min_count=999
    local best_gpu=${GPUS[0]}
    for gpu in "${GPUS[@]}"; do
        # Count training processes using this GPU
        local count=$(ps aux | grep "CUDA_VISIBLE_DEVICES=$gpu python scripts/rsl_rl/train.py" | grep -v grep | wc -l)
        if [ "$count" -lt "$min_count" ]; then
            min_count=$count
            best_gpu=$gpu
        fi
    done
    echo "$best_gpu"
}

# Track PIDs
PIDS=()

launch_job() {
    local cluster_id=$1
    local cluster_name=$2
    local gpu_id=$(find_least_loaded_gpu)
    local registry_name="amass_g1_cluster_${cluster_id}"
    local run_name="cluster_${cluster_id}_${cluster_name}"

    echo "[$(date '+%H:%M:%S')] [Cluster $cluster_id] Training $run_name on GPU $gpu_id"

    CUDA_VISIBLE_DEVICES=$gpu_id python scripts/rsl_rl/train.py \
        --task=Tracking-Flat-G1-v0 \
        --registry_name "wandb-registry-motion/${registry_name}" \
        --headless \
        --logger wandb \
        --log_project_name g1_expert \
        --run_name "${run_name}" &

    PIDS+=($!)
}

# Count alive PIDs
count_running() {
    local count=0
    local alive=()
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            ((count++))
            alive+=($pid)
        fi
    done
    PIDS=("${alive[@]}")
    echo "$count"
}

# Launch initial batch
next_entry=0
for _ in $(seq 1 $MAX_JOBS); do
    if [ $next_entry -ge ${#ENTRIES[@]} ]; then break; fi
    IFS='|' read -r cluster_id cluster_name <<< "${ENTRIES[$next_entry]}"
    launch_job "$cluster_id" "$cluster_name"
    ((next_entry++))
done

# Poll and backfill
while true; do
    sleep $POLL_INTERVAL
    running=$(count_running)

    # Launch new jobs to fill empty slots
    while [ "$running" -lt "$MAX_JOBS" ] && [ $next_entry -lt ${#ENTRIES[@]} ]; do
        IFS='|' read -r cluster_id cluster_name <<< "${ENTRIES[$next_entry]}"
        launch_job "$cluster_id" "$cluster_name"
        ((next_entry++))
        ((running++))
    done

    echo "[$(date '+%H:%M:%S')] Running: $running/$MAX_JOBS, Queued: $(( ${#ENTRIES[@]} - next_entry ))"

    if [ "$running" -eq 0 ] && [ $next_entry -ge ${#ENTRIES[@]} ]; then
        break
    fi
done

echo ""
echo "All cluster expert training jobs completed."
