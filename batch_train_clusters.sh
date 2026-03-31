#!/bin/bash
# Train one BeyondMimic expert per cluster using the representative motion.
#
# Reads cluster_representatives.json to get one motion per cluster,
# then launches parallel training jobs across GPUs.
#
# Prerequisites:
#   1. Run cluster_amass_tmr.py to generate clusters .npz
#   2. Run build_cluster_mapping.py to generate cluster_representatives.json
#   3. Representative motions must be registered in wandb registry
#
# Usage:
#   bash batch_train_clusters.sh <cluster_mapping_dir>
#   bash batch_train_clusters.sh amass_g1/cluster_mapping

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MAPPING_DIR="${1:-amass_g1/cluster_mapping}"
REPS_FILE="$MAPPING_DIR/cluster_representatives.json"
PYTHON="/home/jovyan/conda/beyondmimic-env/bin/python"

if [ ! -f "$REPS_FILE" ]; then
    echo "Error: $REPS_FILE not found"
    echo "Run: python scripts/build_cluster_mapping.py --clusters <clusters.npz> --output_dir $MAPPING_DIR"
    exit 1
fi

GPUS=(0 1 2 3)
JOBS_PER_GPU=2
MAX_JOBS=$((${#GPUS[@]} * JOBS_PER_GPU))

# Parse cluster representatives from JSON
# Extract cluster_id and file path pairs
mapfile -t ENTRIES < <(python -c "
import json
with open('$REPS_FILE') as f:
    reps = json.load(f)
for cid in sorted(reps, key=int):
    # Extract motion name from file path (remove directory and .npz)
    fname = reps[cid]['file']
    print(f'{cid}|{fname}')
")

echo "============================================================"
echo "Training BeyondMimic experts per cluster"
echo "============================================================"
echo "Clusters: ${#ENTRIES[@]}"
echo "GPUs: ${GPUS[*]} (${JOBS_PER_GPU} jobs/GPU = ${MAX_JOBS} parallel)"
echo ""

# Check logs for already-completed trainings
LOG_DIR="logs/rsl_rl/g1_flat"

i=0
for entry in "${ENTRIES[@]}"; do
    IFS='|' read -r cluster_id motion_file <<< "$entry"

    # Derive motion name for wandb registry
    # File path like: g1/ACCAD/.../A14-standtoskip_poses_120_jpos.npz
    # Registry name: amass_g1_<flattened_name>
    basename_no_ext=$(basename "$motion_file" .npz)
    registry_name="amass_g1_${basename_no_ext}"
    run_name="cluster_${cluster_id}_${basename_no_ext}"

    # Check if already trained
    already_done=false
    for d in "$LOG_DIR"/*"$run_name"*; do
        if [ -f "$d/model_29999.pt" ] 2>/dev/null; then
            already_done=true
            break
        fi
    done

    if $already_done; then
        echo "[Cluster $cluster_id] Already trained: $run_name"
        continue
    fi

    gpu_index=$((i % MAX_JOBS))
    gpu_id=${GPUS[$((gpu_index % ${#GPUS[@]}))]}

    echo "[Cluster $cluster_id] Training $run_name on GPU $gpu_id"

    CUDA_VISIBLE_DEVICES=$gpu_id $PYTHON scripts/rsl_rl/train.py \
        --task=Tracking-Flat-G1-v0 \
        --registry_name "wandb-registry-motion/${registry_name}" \
        --headless \
        --logger wandb \
        --log_project_name g1_cluster_experts \
        --run_name "$run_name" &

    ((i++))

    if (( i % MAX_JOBS == 0 )); then
        wait
    fi
done

wait
echo ""
echo "All cluster expert training jobs completed."
