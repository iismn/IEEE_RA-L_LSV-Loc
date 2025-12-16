import os
import glob
import shutil
import subprocess
import tempfile
from pathlib import Path
from tqdm import tqdm
import argparse

def create_video_with_ffmpeg(input_folder, output_path, fps=4, resize=None, quality='medium'):
    """
    FFmpeg를 사용해서 폴더의 이미지들로 비디오 생성
    
    Args:
        input_folder: 이미지가 있는 폴더
        output_path: 출력 비디오 경로
        fps: 초당 프레임 수
        resize: 크기 조절 (예: "1280:720", "50%")
        quality: 품질 ('low', 'medium', 'high')
    """
    # 이미지 파일 찾기
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))
        image_files.extend(glob.glob(os.path.join(input_folder, ext.upper())))
    
    image_files.sort()
    
    if not image_files:
        print(f"No images found in {input_folder}")
        return False
    
    print(f"Found {len(image_files)} images")
    print(f"Creating video: {output_path}")
    
    # 임시 디렉토리 생성
    with tempfile.TemporaryDirectory() as temp_dir:
        print("Copying and renaming images...")
        
        # 이미지들을 임시 디렉토리에 순차적으로 복사
        for i, img_path in enumerate(tqdm(image_files, desc="Copying images")):
            # 확장자 추출
            ext = Path(img_path).suffix
            # 순차적 이름으로 복사
            new_name = f"{i:06d}{ext}"
            new_path = os.path.join(temp_dir, new_name)
            shutil.copy2(img_path, new_path)
        
        # FFmpeg 명령어 구성
        input_pattern = os.path.join(temp_dir, "%06d.png")  # PNG로 가정
        if not glob.glob(os.path.join(temp_dir, "*.png")):
            # PNG가 없으면 첫 번째 파일의 확장자 사용
            first_ext = Path(image_files[0]).suffix
            input_pattern = os.path.join(temp_dir, f"%06d{first_ext}")
        
        # 품질 설정
        quality_settings = {
            'low': ['-crf', '28'],
            'medium': ['-crf', '23'],
            'high': ['-crf', '18']
        }
        
        # FFmpeg 명령어
        cmd = [
            'ffmpeg', '-y',  # -y: 덮어쓰기
            '-framerate', str(fps),
            '-i', input_pattern,
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p'
        ]
        
        # 품질 설정 추가
        cmd.extend(quality_settings.get(quality, quality_settings['medium']))
        
        # 크기 조절
        if resize:
            if '%' in resize:
                # 퍼센트 크기 조절 + 짝수 크기 보장
                scale = f"scale=trunc(iw*{resize.replace('%', '')}/100/2)*2:trunc(ih*{resize.replace('%', '')}/100/2)*2"
            else:
                # 절대 크기 (예: "1280:720")
                scale = f"scale={resize}"
            cmd.extend(['-vf', scale])
        else:
            # 기본적으로 짝수 크기 보장
            cmd.extend(['-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2'])
        
        cmd.append(output_path)
        
        print(f"Running FFmpeg command...")
        print(f"Command: {' '.join(cmd)}")
        
        # FFmpeg 실행
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=temp_dir)
            if result.returncode == 0:
                print(f"Video created successfully: {output_path}")
                print(f"Number of frames: {len(image_files)}")
                print(f"FPS: {fps}")
                print(f"Duration: {len(image_files) / fps:.1f} seconds")
                return True
            else:
                print(f"FFmpeg error: {result.stderr}")
                return False
        except FileNotFoundError:
            print("FFmpeg not found. Please install FFmpeg first.")
            print("Ubuntu/Debian: sudo apt install ffmpeg")
            print("CentOS/RHEL: sudo yum install ffmpeg")
            return False
        except Exception as e:
            print(f"Error running FFmpeg: {e}")
            return False

def create_comparison_video(query_folder, db_folder, output_path, fps=4, quality='medium'):
    """
    Query와 Database 이미지를 나란히 배치한 비교 영상 생성
    """
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
        return False
    
    # 파일 개수 맞추기
    min_count = min(len(query_files), len(db_files))
    query_files = query_files[:min_count]
    db_files = db_files[:min_count]
    
    print(f"Creating comparison video with {min_count} frame pairs")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        query_dir = os.path.join(temp_dir, "query")
        db_dir = os.path.join(temp_dir, "db")
        os.makedirs(query_dir)
        os.makedirs(db_dir)
        
        print("Preparing query images...")
        for i, img_path in enumerate(tqdm(query_files, desc="Query images")):
            ext = Path(img_path).suffix
            new_name = f"{i:06d}{ext}"
            shutil.copy2(img_path, os.path.join(query_dir, new_name))
        
        print("Preparing database images...")
        for i, img_path in enumerate(tqdm(db_files, desc="DB images")):
            ext = Path(img_path).suffix
            new_name = f"{i:06d}{ext}"
            shutil.copy2(img_path, os.path.join(db_dir, new_name))
        
        # FFmpeg로 나란히 배치한 비디오 생성
        query_pattern = os.path.join(query_dir, "%06d.png")
        if not glob.glob(os.path.join(query_dir, "*.png")):
            first_ext = Path(query_files[0]).suffix
            query_pattern = os.path.join(query_dir, f"%06d{first_ext}")
        
        db_pattern = os.path.join(db_dir, "%06d.png")
        if not glob.glob(os.path.join(db_dir, "*.png")):
            first_ext = Path(db_files[0]).suffix
            db_pattern = os.path.join(db_dir, f"%06d{first_ext}")
        
        # 품질 설정
        quality_settings = {
            'low': ['-crf', '28'],
            'medium': ['-crf', '23'],
            'high': ['-crf', '18']
        }
        
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', query_pattern,
            '-framerate', str(fps),
            '-i', db_pattern,
            '-filter_complex', 
            '[0:v]scale=640:-1,pad=1280:720:0:(oh-ih)/2:black,drawtext=text="Query":fontsize=30:fontcolor=white:x=10:y=10[left];'
            '[1:v]scale=640:-1,pad=1280:720:0:(oh-ih)/2:black,drawtext=text="Database":fontsize=30:fontcolor=white:x=10:y=10[right];'
            '[left][right]hstack=inputs=2[v]',
            '-map', '[v]',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p'
        ]
        
        # 품질 설정 추가
        cmd.extend(quality_settings.get(quality, quality_settings['medium']))
        cmd.append(output_path)
        
        print("Creating comparison video...")
        print(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"Comparison video created: {output_path}")
                return True
            else:
                print(f"FFmpeg error: {result.stderr}")
                return False
        except Exception as e:
            print(f"Error creating comparison video: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Create videos from images using FFmpeg')
    parser.add_argument('--base_dir', type=str, required=True, help='Base directory containing visualization folders')
    parser.add_argument('--fps', type=float, default=4.0, help='Frames per second')
    parser.add_argument('--resize', type=str, help='Resize option (e.g., "50%", "1280:720")')
    parser.add_argument('--quality', type=str, choices=['low', 'medium', 'high'], default='medium', help='Video quality')
    parser.add_argument('--comparison_only', action='store_true', help='Create only comparison video')
    
    args = parser.parse_args()
    
    base_dir = args.base_dir
    vis_dir = os.path.join(base_dir, "visualizations")
    
    if not os.path.exists(vis_dir):
        print(f"Visualizations directory not found: {vis_dir}")
        return
    
    # 비디오 출력 디렉토리 생성
    video_dir = os.path.join(base_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)
    
    if args.comparison_only:
        # 비교 영상만 생성
        query_attention_dir = os.path.join(vis_dir, "attention_overlays", "query")
        db_attention_dir = os.path.join(vis_dir, "attention_overlays", "database")
        
        if os.path.exists(query_attention_dir) and os.path.exists(db_attention_dir):
            comparison_path = os.path.join(video_dir, "attention_comparison.mp4")
            create_comparison_video(
                query_attention_dir,
                db_attention_dir,
                comparison_path,
                fps=args.fps,
                quality=args.quality
            )
    else:
        # 개별 비디오들 생성
        folders_to_process = [
            ("attention_overlays/query", "query_attention_overlays.mp4"),
            ("attention_overlays/database", "database_attention_overlays.mp4"), 
            ("feature_matches", "feature_matches.mp4")
        ]
        
        for folder_path, output_name in folders_to_process:
            input_dir = os.path.join(vis_dir, folder_path)
            if os.path.exists(input_dir):
                output_path = os.path.join(video_dir, output_name)
                create_video_with_ffmpeg(
                    input_dir,
                    output_path,
                    fps=args.fps,
                    resize=args.resize,
                    quality=args.quality
                )
                print()  # 빈 줄 추가
    
    print(f"Videos saved in: {video_dir}")

if __name__ == "__main__":
    main()
