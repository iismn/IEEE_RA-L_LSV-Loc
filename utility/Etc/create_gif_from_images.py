import os
import cv2
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
from pathlib import Path

def create_gif_from_folder(input_folder, output_path, duration=500, max_images=None, resize_factor=0.5):
    """
    폴더 내의 이미지들로 GIF를 생성
    
    Args:
        input_folder: 이미지가 있는 폴더 경로
        output_path: 출력할 GIF 파일 경로
        duration: 각 프레임의 지속시간 (밀리초)
        max_images: 최대 이미지 개수 (None이면 모든 이미지)
        resize_factor: 이미지 크기 조절 비율
    """
    # 지원하는 이미지 확장자
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
    
    # 모든 이미지 파일 찾기
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))
        image_files.extend(glob.glob(os.path.join(input_folder, ext.upper())))
    
    # 파일명으로 정렬
    image_files.sort()
    
    if not image_files:
        print(f"No images found in {input_folder}")
        return
    
    # 최대 이미지 개수 제한
    if max_images and len(image_files) > max_images:
        # 균등하게 샘플링
        step = len(image_files) // max_images
        image_files = image_files[::step][:max_images]
    
    print(f"Found {len(image_files)} images")
    print(f"Creating GIF: {output_path}")
    
    # 이미지 읽기 및 처리
    pil_images = []
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            # PIL로 이미지 읽기
            img = Image.open(img_path)
            
            # RGB로 변환 (필요한 경우)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 크기 조절
            if resize_factor != 1.0:
                new_width = int(img.width * resize_factor)
                new_height = int(img.height * resize_factor)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            pil_images.append(img)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    if not pil_images:
        print("No valid images found!")
        return
    
    # GIF 생성
    try:
        pil_images[0].save(
            output_path,
            save_all=True,
            append_images=pil_images[1:],
            duration=duration,
            loop=0,
            optimize=True
        )
        print(f"GIF saved successfully: {output_path}")
        print(f"Number of frames: {len(pil_images)}")
        print(f"Frame duration: {duration}ms")
        print(f"Total duration: {len(pil_images) * duration / 1000:.1f}s")
    except Exception as e:
        print(f"Error creating GIF: {e}")

def main():
    parser = argparse.ArgumentParser(description='Create GIF from images in folders')
    parser.add_argument('--base_dir', type=str, required=True, help='Base directory containing visualization folders')
    parser.add_argument('--duration', type=int, default=800, help='Duration per frame in milliseconds')
    parser.add_argument('--max_images', type=int, default=50, help='Maximum number of images per GIF')
    parser.add_argument('--resize_factor', type=float, default=0.3, help='Resize factor for images (0.5 = half size)')
    
    args = parser.parse_args()
    
    base_dir = args.base_dir
    
    # 시각화 폴더 구조 확인
    vis_dir = os.path.join(base_dir, "visualizations")
    
    if not os.path.exists(vis_dir):
        print(f"Visualizations directory not found: {vis_dir}")
        return
    
    # GIF 출력 디렉토리 생성
    gif_dir = os.path.join(base_dir, "gifs")
    os.makedirs(gif_dir, exist_ok=True)
    
    # 1. Query attention overlays GIF
    query_attention_dir = os.path.join(vis_dir, "attention_overlays", "query")
    if os.path.exists(query_attention_dir):
        query_gif_path = os.path.join(gif_dir, "query_attention_overlays.gif")
        create_gif_from_folder(
            query_attention_dir, 
            query_gif_path, 
            duration=args.duration,
            max_images=args.max_images,
            resize_factor=args.resize_factor
        )
    
    # 2. Database attention overlays GIF
    db_attention_dir = os.path.join(vis_dir, "attention_overlays", "database")
    if os.path.exists(db_attention_dir):
        db_gif_path = os.path.join(gif_dir, "database_attention_overlays.gif")
        create_gif_from_folder(
            db_attention_dir, 
            db_gif_path, 
            duration=args.duration,
            max_images=args.max_images,
            resize_factor=args.resize_factor
        )
    
    # 3. Feature matches GIF
    feature_matches_dir = os.path.join(vis_dir, "feature_matches")
    if os.path.exists(feature_matches_dir):
        matches_gif_path = os.path.join(gif_dir, "feature_matches.gif")
        create_gif_from_folder(
            feature_matches_dir, 
            matches_gif_path, 
            duration=args.duration,
            max_images=args.max_images,
            resize_factor=args.resize_factor
        )
    
    # 4. Range image matches GIF (만약 있다면)
    range_dir = os.path.join(vis_dir, "range_matches")
    if os.path.exists(range_dir):
        range_gif_path = os.path.join(gif_dir, "range_matches.gif")
        create_gif_from_folder(
            range_dir, 
            range_gif_path, 
            duration=args.duration,
            max_images=args.max_images,
            resize_factor=args.resize_factor
        )
    
    print(f"\nAll GIFs saved in: {gif_dir}")

if __name__ == "__main__":
    main()
