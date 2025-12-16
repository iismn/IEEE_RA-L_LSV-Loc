#!/bin/bash

# GIF 생성 스크립트
# Usage: ./create_gifs.sh [experiment_directory]

# 실험 디렉토리 설정 (기본값: result/04160633_dinov2_vitb14_reg)
EXPERIMENT_DIR=${1:-"result/04160633_dinov2_vitb14_reg"}

echo "Creating GIFs from images in: $EXPERIMENT_DIR"

python create_gif_from_images.py \
    --base_dir "$EXPERIMENT_DIR" \
    --duration 800 \
    --max_images 50 \
    --resize_factor 0.3

echo "Done! Check the gifs folder in $EXPERIMENT_DIR"
