#!/bin/bash

# 비디오 생성 스크립트
# Usage: ./create_videos.sh [experiment_directory] [options]

# 실험 디렉토리 설정 (기본값: result/04160633_dinov2_vitb14_reg)
EXPERIMENT_DIR=${1:-"result/04160633_dinov2_vitb14_reg"}

echo "Creating videos from images in: $EXPERIMENT_DIR"

echo "1. Creating individual videos..."
python create_video_from_images.py \
    --base_dir "$EXPERIMENT_DIR" \
    --fps 3.0 \
    --resize_factor 0.8

echo "2. Creating side-by-side comparison video..."
python create_video_from_images.py \
    --base_dir "$EXPERIMENT_DIR" \
    --fps 3.0 \
    --resize_factor 0.6 \
    --side_by_side

echo "Done! Check the videos folder in $EXPERIMENT_DIR"
