#!/bin/bash
# Batch preprocess AMASS G1 motions: npz_to_npz.py (FK) -> filter_motions.py -> cluster
#
# Step 1: Run npz_to_npz.py on all amass_g1 npz files (multi-GPU parallel)
# Step 2: Run filter_motions.py on processed output
# Step 3: Cluster filtered motions
#
# Usage:
#   bash batch_preprocess_amass.sh                        # Run all steps (default: kinematic cluster)
#   bash batch_preprocess_amass.sh --step 1               # Run only step 1 (preprocess)
#   bash batch_preprocess_amass.sh --step 2               # Run only step 2 (filter)
#   bash batch_preprocess_amass.sh --step 3               # Run only step 3 (cluster)
#   bash batch_preprocess_amass.sh --cluster tmr          # Use TMR clustering
#   bash batch_preprocess_amass.sh --cluster kinematic    # Use kinematic clustering (default)
#   bash batch_preprocess_amass.sh --step 3 --cluster tmr # Step 3 only with TMR

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AMASS_DIR="$SCRIPT_DIR/amass_g1/g1"
PROCESSED_DIR="$SCRIPT_DIR/amass_g1/processed/g1"
GPUS=(0 1 2 3)
JOBS_PER_GPU=2
MAX_JOBS=$((${#GPUS[@]} * JOBS_PER_GPU))
PYTHON="/home/jovyan/conda/beyondmimic-env/bin/python"

STEP="all"
CLUSTER_METHOD="kinematic"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --step)     STEP="$2"; shift 2;;
        --cluster)  CLUSTER_METHOD="$2"; shift 2;;
        *)          STEP="$1"; shift;;
    esac
done

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
# Step 3: Clustering
# ============================================================
if [[ "$STEP" == "all" || "$STEP" == "3" ]]; then
    FILTERED_DIR="$SCRIPT_DIR/amass_g1/filtered/g1"

    if [ ! -d "$FILTERED_DIR" ]; then
        echo "Filtered directory not found: $FILTERED_DIR"
        echo "Run step 2 first."
        exit 1
    fi

    if [[ "$CLUSTER_METHOD" == "tmr" ]]; then
        echo "============================================================"
        echo "Step 3: Clustering filtered motions (TMR)"
        echo "============================================================"

        $PYTHON scripts/cluster_amass_tmr.py \
            --amass_dir "$FILTERED_DIR" \
            --density_subsample 5000 \
            --output "$SCRIPT_DIR/amass_g1/clusters.npz"

        echo ""
        echo "Building cluster mapping..."
        $PYTHON scripts/build_cluster_mapping.py \
            --clusters "$SCRIPT_DIR/amass_g1/clusters.npz" \
            --output_dir "$SCRIPT_DIR/amass_g1/cluster_mapping"

    elif [[ "$CLUSTER_METHOD" == "kinematic" ]]; then
        echo "============================================================"
        echo "Step 3: Clustering processed motions (Kinematic K-Means)"
        echo "============================================================"

        $PYTHON scripts/cluster_amass_kinematic.py \
            --amass_dir "$PROCESSED_DIR" \
            --k 20 \
            --output "$SCRIPT_DIR/amass_g1/clusters_kinematic.npz"

        echo ""
        echo "Building cluster mapping..."
        $PYTHON scripts/build_cluster_mapping.py \
            --clusters "$SCRIPT_DIR/amass_g1/clusters_kinematic.npz" \
            --output_dir "$SCRIPT_DIR/amass_g1/cluster_mapping_kinematic"

    else
        echo "Unknown cluster method: $CLUSTER_METHOD (use 'tmr' or 'kinematic')"
        exit 1
    fi

    echo "Step 3 complete!"
    echo ""
fi

echo "Done!"
