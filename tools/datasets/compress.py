import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

cpu_cores = os.cpu_count()

def convert_video(src_path, dest_path):
    # 如果目标文件存在，跳过转换
    if os.path.exists(dest_path):
        return
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    ffmpeg_path = '/home/fangchen/miniconda3/bin/ffmpeg'
    cmd = [
        ffmpeg_path,
        '-y',  # 覆盖输出文件
        '-i', src_path,
        '-vf', 'scale=-2:480',
        '-r', '24',
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-c:a', 'copy',
        dest_path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print(f"Error converting {src_path}:\n{result.stderr}")
    except Exception as e:
        print(f"Exception occurred while converting {src_path}:\n{e}")

def copy_file(src_path, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    try:
        shutil.copy2(src_path, dest_path)
    except Exception as e:
        print(f"Error copying {src_path} to {dest_path}:\n{e}")

def process_files(src_dir, dest_dir):
    video_files = []
    other_files = []
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            src_file_path = os.path.join(root, file)
            rel_path = os.path.relpath(src_file_path, src_dir)
            dest_file_path = os.path.join(dest_dir, rel_path)
            if file.lower().endswith('.mp4'):
                video_files.append((src_file_path, dest_file_path))
            else:
                other_files.append((src_file_path, dest_file_path))

    # 复制其他文件
    for src_path, dest_path in other_files:
        copy_file(src_path, dest_path)

    # 使用多进程转换视频，显示进度条
    with ProcessPoolExecutor(max_workers=cpu_cores/2) as executor:
        futures = [executor.submit(convert_video, src_path, dest_path) for src_path, dest_path in video_files]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Converting videos", unit="video"):
            future.result()

if __name__ == '__main__':
    src_directory = '/shared/fangchen/junrui/Open-Sora/dataset/droid_raw'
    dest_directory = '/shared/fangchen/junrui/Open-Sora/dataset/droid_480p24fps'
    process_files(src_directory, dest_directory)
