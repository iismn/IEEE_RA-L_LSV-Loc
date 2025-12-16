import os
import cv2
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
from pathlib import Path

def create_video_from_folder(input_folder, output_path, fps=2, resize_factor=1.0, video_codec='mp4v'):
    """
    폴더 내의 모든 이미지들로 영상을 생성
    
    Args:
        input_folder: 이미지가 있는 폴더 경로
        output_path: 출력할 영상 파일 경로
        fps: 초당 프레임 수
        resize_factor: 이미지 크기 조절 비율
        video_codec: 비디오 코덱 ('mp4v', 'XVID', 'MJPG' 등)
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
    
    print(f"Found {len(image_files)} images")
    print(f"Creating video: {output_path}")
    
    # 첫 번째 이미지로 비디오 크기 결정
    first_img = cv2.imread(image_files[0])
    if first_img is None:
        print(f"Cannot read first image: {image_files[0]}")
        return
    
    height, width = first_img.shape[:2]
    if resize_factor != 1.0:
        width = int(width * resize_factor)
        height = int(height * resize_factor)
    
    # VideoWriter 설정 - 다양한 코덱 시도
    codecs_to_try = ['XVID', 'MJPG', 'mp4v', 'DIVX']
    video_writer = None
    
    for codec in codecs_to_try:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if video_writer.isOpened():
                print(f"Using codec: {codec}")
                break
            else:
                video_writer.release()
        except:
            continue
    
    if video_writer is None or not video_writer.isOpened():
        print(f"Error: Could not open video writer with any codec for {output_path}")
        # AVI 형식으로 시도
        output_path = output_path.replace('.mp4', '.avi')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not video_writer.isOpened():
            print(f"Failed to create video with AVI format as well")
            return
        else:
            print(f"Using AVI format: {output_path}")
    
    # 이미지들을 비디오에 추가
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            # 이미지 읽기
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read {img_path}")
                continue
            
            # 크기 조절
            if resize_factor != 1.0:
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
            
            # 비디오에 프레임 추가
            video_writer.write(img)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    # 리소스 해제
    video_writer.release()
    
    print(f"Video saved successfully: {output_path}")
    print(f"Number of frames: {len(image_files)}")
    print(f"FPS: {fps}")
    print(f"Duration: {len(image_files) / fps:.1f} seconds")
    print(f"Resolution: {width}x{height}")

def create_side_by_side_video(query_folder, db_folder, output_path, fps=2, resize_factor=0.8):
    """
    Query와 Database 이미지를 나란히 배치한 영상 생성
    """
    # 이미지 파일들 찾기
    def get_image_files(folder):
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
        files = []
        for ext in image_extensions:
            files.extend(glob.glob(os.path.join(folder, ext)))
            files.extend(glob.glob(os.path.join(folder, ext.upper())))
        return sorted(files)
    
    query_files = get_image_files(query_folder)
    db_files = get_image_files(db_folder)
    
    if not query_files or not db_files:
        print("No images found in one or both folders")
        return
    
    # 파일 개수 맞추기 (적은 쪽에 맞춤)
    min_count = min(len(query_files), len(db_files))
    query_files = query_files[:min_count]
    db_files = db_files[:min_count]
    
    print(f"Creating side-by-side video with {min_count} frame pairs")
    print(f"Output: {output_path}")
    
    # 첫 번째 이미지들로 크기 결정
    query_img = cv2.imread(query_files[0])
    db_img = cv2.imread(db_files[0])
    
    if query_img is None or db_img is None:
        print("Cannot read first images")
        return
    
    # 이미지 크기 조절
    q_height, q_width = query_img.shape[:2]
    d_height, d_width = db_img.shape[:2]
    
    # 같은 높이로 맞추기
    target_height = int(min(q_height, d_height) * resize_factor)
    q_new_width = int(q_width * target_height / q_height)
    d_new_width = int(d_width * target_height / d_height)
    
    # 최종 비디오 크기
    final_width = q_new_width + d_new_width
    final_height = target_height
    
    # VideoWriter 설정 - 다양한 코덱 시도
    codecs_to_try = ['XVID', 'MJPG', 'mp4v', 'DIVX']
    video_writer = None
    
    for codec in codecs_to_try:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (final_width, final_height))
            if video_writer.isOpened():
                print(f"Using codec: {codec}")
                break
            else:
                video_writer.release()
        except:
            continue
    
    if video_writer is None or not video_writer.isOpened():
        print(f"Error: Could not open video writer with any codec for {output_path}")
        # AVI 형식으로 시도
        output_path = output_path.replace('.mp4', '.avi')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (final_width, final_height))
        if not video_writer.isOpened():
            print(f"Failed to create video with AVI format as well")
            return
        else:
            print(f"Using AVI format: {output_path}")
    
    # 이미지 쌍들을 처리
    for q_path, d_path in tqdm(zip(query_files, db_files), total=min_count, desc="Processing image pairs"):
        try:
            # 이미지 읽기
            q_img = cv2.imread(q_path)
            d_img = cv2.imread(d_path)
            
            if q_img is None or d_img is None:
                continue
            
            # 크기 조절
            q_img = cv2.resize(q_img, (q_new_width, target_height))
            d_img = cv2.resize(d_img, (d_new_width, target_height))
            
            # 나란히 배치
            combined = np.hstack([q_img, d_img])
            
            # 구분선 그리기
            cv2.line(combined, (q_new_width, 0), (q_new_width, target_height), (255, 255, 255), 2)
            
            # 텍스트 추가
            cv2.putText(combined, "Query", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined, "Database", (q_new_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # 비디오에 프레임 추가
            video_writer.write(combined)
            
        except Exception as e:
            print(f"Error processing {q_path}, {d_path}: {e}")
            continue
    
    # 리소스 해제
    video_writer.release()
    
    print(f"Side-by-side video saved: {output_path}")
    print(f"Resolution: {final_width}x{final_height}")
    print(f"Duration: {min_count / fps:.1f} seconds")

def main():
    parser = argparse.ArgumentParser(description='Create videos from images in folders')
    parser.add_argument('--base_dir', type=str, required=True, help='Base directory containing visualization folders')
    parser.add_argument('--fps', type=float, default=2.0, help='Frames per second')
    parser.add_argument('--resize_factor', type=float, default=0.8, help='Resize factor for images')
    parser.add_argument('--side_by_side', action='store_true', help='Create side-by-side comparison videos')
    
    args = parser.parse_args()
    
    base_dir = args.base_dir
    
    # 시각화 폴더 구조 확인
    vis_dir = os.path.join(base_dir, "visualizations")
    
    if not os.path.exists(vis_dir):
        print(f"Visualizations directory not found: {vis_dir}")
        return
    
    # 비디오 출력 디렉토리 생성
    video_dir = os.path.join(base_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)
    
    if args.side_by_side:
        # Query vs Database 비교 영상 생성
        query_attention_dir = os.path.join(vis_dir, "attention_overlays", "query")
        db_attention_dir = os.path.join(vis_dir, "attention_overlays", "database")
        
        if os.path.exists(query_attention_dir) and os.path.exists(db_attention_dir):
            comparison_video_path = os.path.join(video_dir, "attention_comparison.mp4")
            create_side_by_side_video(
                query_attention_dir,
                db_attention_dir,
                comparison_video_path,
                fps=args.fps,
                resize_factor=args.resize_factor
            )
    else:
        # 개별 비디오 생성
        # 1. Query attention overlays 비디오
        query_attention_dir = os.path.join(vis_dir, "attention_overlays", "query")
        if os.path.exists(query_attention_dir):
            query_video_path = os.path.join(video_dir, "query_attention_overlays.mp4")
            create_video_from_folder(
                query_attention_dir, 
                query_video_path, 
                fps=args.fps,
                resize_factor=args.resize_factor
            )
        
        # 2. Database attention overlays 비디오
        db_attention_dir = os.path.join(vis_dir, "attention_overlays", "database")
        if os.path.exists(db_attention_dir):
            db_video_path = os.path.join(video_dir, "database_attention_overlays.mp4")
            create_video_from_folder(
                db_attention_dir, 
                db_video_path, 
                fps=args.fps,
                resize_factor=args.resize_factor
            )
        
        # 3. Feature matches 비디오
        feature_matches_dir = os.path.join(vis_dir, "feature_matches")
        if os.path.exists(feature_matches_dir):
            matches_video_path = os.path.join(video_dir, "feature_matches.mp4")
            create_video_from_folder(
                feature_matches_dir, 
                matches_video_path, 
                fps=args.fps,
                resize_factor=args.resize_factor
            )
    
    print(f"\nAll videos saved in: {video_dir}")

if __name__ == "__main__":
    main()
