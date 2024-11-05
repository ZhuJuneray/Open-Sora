import csv
import h5py
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np

def calculate_ffmpeg_timestamps(length, target_fps=3, source_fps=60):
    """
    计算与ffmpeg相匹配的采样时间戳
    参数:
    length: 原始序列长度
    target_fps: 目标帧率
    source_fps: 源帧率
    """
    # 计算源视频的总时长（秒）
    duration = (length - 1) / source_fps
    
    # 生成目标视频的时间戳（秒）
    target_timestamps = np.arange(0, duration, 1/target_fps)
    
    # 将时间戳转换为源视频的帧索引
    indices = np.round(target_timestamps * source_fps).astype(int)
    
    # 确保不超过源视频长度
    indices = indices[indices < length]
    
    return indices

def hdf5_to_json(video_file_path, hdf5_file_path, json_file_path, episode_id, texts):
    """Convert specific parts of an HDF5 file to a JSON file."""
    with h5py.File(hdf5_file_path, 'r') as file:
        # 提取数据
        cartesian_position = file['observation']['robot_state']['cartesian_position'][()]
        gripper_position = file['observation']['robot_state']['gripper_position'][()]
        
        # 获取采样索引
        # 由于action比video多一帧，使用video的长度来计算
        video_length = len(cartesian_position) - 1
        indices = calculate_ffmpeg_timestamps(video_length)
        
        # 打印调试信息
        print(f"Original length: {len(cartesian_position)}")
        print(f"Video length (original - 1): {video_length}")
        print(f"Sampled indices: {indices}")
        print(f"Number of sampled frames: {len(indices)}")
        
        # 使用计算出的索引进行采样
        cartesian_position_3fps = cartesian_position[indices]
        gripper_position_3fps = gripper_position[indices]

        data_dict = {
            "task": "robot_trajectory_prediction",
            "texts": [texts],
            "videos": [{
                "video_path": video_file_path.replace(
                    '/shared/fangchen/junrui/Open-Sora/dataset/droid_480p3fps/',
                    'videos/droid_480p3fps/'
                )
            }],
            "state": cartesian_position_3fps.tolist(),
            "continuous_gripper_state": gripper_position_3fps.tolist(),
            "episode_id": str(episode_id)
        }

        # 写入JSON文件
        with open(json_file_path, 'w') as json_file:
            json.dump(data_dict, json_file, indent=2)

def process_row(row, episode_id):
    """Process a single row of the CSV."""
    video_file_path = row['path']
    hdf5_file_path = Path(
        row['path'].replace('/recordings/MP4', '').replace('/droid_480p3fps/', '/droid_raw_action/')
    ).with_name('trajectory.h5')
    json_file_path = f'dataset/opensource_robotdata/droid/annotation/{episode_id}.json'
    texts = row['text']
    
    hdf5_to_json(video_file_path, hdf5_file_path, json_file_path, episode_id, texts)

def process_csv(csv_file_path):
    """Process each row of a CSV file with multithreading."""
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = list(csv.DictReader(csv_file))
        
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(process_row, row, episode_id): episode_id 
                      for episode_id, row in enumerate(csv_reader)}
            
            for _ in tqdm(as_completed(futures), total=len(futures), 
                         desc="Processing CSV rows", unit="row"):
                pass

# Example usage
csv_file_path = 'dataloader/480p3fps/droid_480p3fps.csv'
process_csv(csv_file_path)


# single thread version (original)
# import csv
# import h5py
# import json
# import os
# from pathlib import Path

# def hdf5_to_json(video_file_path, hdf5_file_path, json_file_path, episode_id, texts):
#     """Convert specific parts of an HDF5 file to a JSON file."""
#     with h5py.File(hdf5_file_path, 'r') as file:
#         # Extract specific datasets
#         cartesian_position = file['observation']['robot_state']['cartesian_position'][()]
#         gripper_position = file['observation']['robot_state']['gripper_position'][()]

#         # Convert 60fps data to 3fps by selecting every 20th frame
#         cartesian_position_3fps = cartesian_position[::20]
#         gripper_position_3fps = gripper_position[::20]

#         data_dict = {
#             "task": "robot_trajectory_prediction",
#             "texts": [texts],
#             "videos": [{
#                 "video_path": video_file_path.replace(
#                     '/shared/fangchen/junrui/Open-Sora/dataset/droid_480p3fps/',
#                     'videos/droid_480p3fps/'
#                 )
#             }],
#             "state": cartesian_position_3fps.tolist(),
#             "continuous_gripper_state": gripper_position_3fps.tolist(),
#             "episode_id": str(episode_id)
#         }

#         # Write JSON to file
#         with open(json_file_path, 'w') as json_file:
#             json.dump(data_dict, json_file, indent=2)

#         print(f"Converted {hdf5_file_path} to {json_file_path}")

# def process_csv(csv_file_path):
#     """Process each row of a CSV file."""
#     with open(csv_file_path, 'r') as csv_file:
#         csv_reader = csv.DictReader(csv_file)
#         episode_id = 0
        
#         for row in csv_reader:
#             # Construct file paths
#             video_file_path = row['path']
#             hdf5_file_path = Path(
#                 row['path'].replace('/recordings/MP4', '').replace('/droid_480p3fps/', '/droid_raw_action/')
#             ).with_name('trajectory.h5')
#             json_file_path = f'dataset/opensource_robotdata/droid/annotation/{episode_id}.json'
#             texts = row['text']
            
#             # Convert HDF5 to JSON
#             hdf5_to_json(video_file_path, hdf5_file_path, json_file_path, episode_id, texts)
            
#             episode_id += 1

# # Example usage
# csv_file_path = 'dataloader/480p3fps/droid_480p3fps.csv'
# process_csv(csv_file_path)
