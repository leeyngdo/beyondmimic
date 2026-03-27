#!/bin/bash

TARGET_DIR="/home/jovyan/whole_body_tracking/LAFAN1_Retargeting_Dataset/g1"

for file in "$TARGET_DIR"/*.csv; do
    # 파일명에서 확장자 제거
    motion_name=$(basename "$file" .csv)

    echo "Processing $motion_name..."

    python scripts/csv_to_npz.py \
        --input_file "$file" \
        --input_fps 30 \
        --output_name "$motion_name" \
        --headless
done