#!/bin/bash
# Batch preprocess AMASS G1 motions: npz_to_npz.py (FK) -> filter_motions.py -> cluster
#
# Step 1: Run npz_to_npz.py on all amass_g1 npz files (multi-GPU parallel)
# Step 2: Run filter_motions.py on processed output
# Step 3: Run cluster_amass_tmr.py on filtered output
#
# Usage:
#   bash batch_preprocess_amass.sh          # Run all steps
#   bash batch_preprocess_amass.sh --step 1 # Run only step 1 (preprocess)
#   bash batch_preprocess_amass.sh --step 2 # Run only step 2 (filter)
#   bash batch_preprocess_amass.sh --step 3 # Run only step 3 (cluster)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AMASS_DIR="$SCRIPT_DIR/amass_g1/g1"
PROCESSED_DIR="$SCRIPT_DIR/amass_g1/processed/g1"
GPUS=(0 1 2 3)
JOBS_PER_GPU=2
MAX_JOBS=$((${#GPUS[@]} * JOBS_PER_GPU))
PYTHON="/home/jovyan/conda/beyondmimic-env/bin/python"

STEP="${1:-all}"
if [[ "$1" == "--step" ]]; then
    STEP="$2"
fi

# ============================================================
# Step 1: npz_to_npz.py (FK via IsaacSim)
# ============================================================
if [[ "$STEP" == "all" || "$STEP" == "1" ]]; then
    echo "============================================================"
    echo "Step 1: Preprocessing AMASS motions (npz_to_npz.py)"
    echo "============================================================"

    # Collect all npz files
    mapfile -t ALL_FILES < <(find "$AMASS_DIR" -name "*.npz" | sort)
    echo "Total files: ${#ALL_FILES[@]}"

    # Find files not yet processed
    TODO=()
    for file in "${ALL_FILES[@]}"; do
        # Build output name: preserve dataset/subject structure
        rel_path="${file#$AMASS_DIR/}"
        # Replace / with _ to create flat output name, remove .npz
        output_name="${rel_path//\//_}"
        output_name="${output_name%.npz}"
        output_file="$PROCESSED_DIR/${output_name}.npz"

        if [ ! -f "$output_file" ]; then
            TODO+=("$file|$output_name")
        fi
    done

    echo "Already processed: $(( ${#ALL_FILES[@]} - ${#TODO[@]} ))"
    echo "Remaining: ${#TODO[@]}"
    echo ""

    if [ ${#TODO[@]} -eq 0 ]; then
        echo "All files already processed!"
    else
        i=0
        for entry in "${TODO[@]}"; do
            IFS='|' read -r file output_name <<< "$entry"

            gpu_index=$((i % MAX_JOBS))
            gpu_id=${GPUS[$((gpu_index % ${#GPUS[@]}))]}

            echo "[$((i+1))/${#TODO[@]}] $output_name on GPU $gpu_id"

            CUDA_VISIBLE_DEVICES=$gpu_id $PYTHON scripts/npz_to_npz.py \
                --input_file "$file" \
                --output_name "$output_name" \
                --robot g1 \
                --headless &

            ((i++))

            if (( i % MAX_JOBS == 0 )); then
                wait
            fi
        done
        wait
        echo "Step 1 complete!"
    fi
    echo ""
fi

# ============================================================
# Step 2: filter_motions.py
# ============================================================
if [[ "$STEP" == "all" || "$STEP" == "2" ]]; then
    echo "============================================================"
    echo "Step 2: Filtering motions (filter_motions.py)"
    echo "============================================================"

    DATASET_DIR="$SCRIPT_DIR" $PYTHON scripts/filter_motions.py \
        --dataset amass_g1 \
        --robot g1

    echo "Step 2 complete!"
    echo ""
fi

# ============================================================
# Step 3: cluster_amass_tmr.py
# ============================================================
if [[ "$STEP" == "all" || "$STEP" == "3" ]]; then
    echo "============================================================"
    echo "Step 3: Clustering filtered motions (cluster_amass_tmr.py)"
    echo "============================================================"

    FILTERED_DIR="$SCRIPT_DIR/amass_g1/filtered/g1"

    if [ ! -d "$FILTERED_DIR" ]; then
        echo "Filtered directory not found: $FILTERED_DIR"
        echo "Run step 2 first."
        exit 1
    fi

    $PYTHON scripts/cluster_amass_tmr.py \
        --amass_dir "$FILTERED_DIR" \
        --density_subsample 5000 \
        --output "$SCRIPT_DIR/amass_g1/clusters.npz"

    echo "Step 3 complete!"
    echo ""
fi

echo "Done!"
