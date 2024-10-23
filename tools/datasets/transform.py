import argparse
import os
import random
import base64
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
from tqdm import tqdm

from .utils import IMG_EXTENSIONS, extract_frames

tqdm.pandas()

try:
    from pandarallel import pandarallel

    pandarallel.initialize(progress_bar=True)
    pandas_has_parallel = True
except ImportError:
    pandas_has_parallel = False


def apply(df, func, **kwargs):
    if pandas_has_parallel:
        return df.parallel_apply(func, **kwargs)
    return df.progress_apply(func, **kwargs)


def get_new_path(path, input_dir, output):
    path_new = os.path.join(output, os.path.relpath(path, input_dir))
    os.makedirs(os.path.dirname(path_new), exist_ok=True)
    return path_new


def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def generate_jsonl_entry(video_path, frames, task_description):
    # Encode the available frames to base64
    base64_images = [image_to_base64(frame) for frame in frames]
    
    # 检测重复帧，如果有重复，返回 None
    if len(set(base64_images)) != len(frames):
        print(f"Warning: Duplicate frames detected in the video: {video_path}")
        return None  # 跳过保存重复帧的条目
    
    message_content = \
    f"""Generate a synthetic description for this video in a very detailed manner based on the 4 uniformaly extracted frames and a high-level task description from the dataset. 
    Do not describe it frame by frame, instead, unified narrative that captures the overall task.
    Pay attention to all objects and its texture and the environmrnt in the video. The camera view is either on robot arm frame, or world frame, which means the camera is either fixed to a robotic arm (which moves relative to the world) or stationary relative to the world. 
    The description should be useful for AI to re-generate the video. 
    The description should be no more than four sentences. 
    For example, a good description would be: 
    "
    A grey robot arm succeeds in moving a red block from the left side of the table to the right side of the table in a well lighted lab. The table is made of wood and the block is made of plastic. The robot arm is moving slowly and carefully to avoid dropping the block. The camera view is fixed to the robot arm frame.
    "

    Now, the High-Level Task Description for this video is:
    “{task_description}”
    Your refined description here:
    """
                        
    # Dynamically add image URLs based on the number of available frames
    image_entries = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}} for base64_image in base64_images]

    jsonl_entry = {
        "custom_id": video_path,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": message_content,
                        },
                        image_entries[0],
                        image_entries[1],
                        image_entries[2],
                        image_entries[3],
                    ],
                },
            ],
            "temperature": 0,
            "max_tokens": 1000
        }
    }

    return jsonl_entry


def process_video(data_line, output_dir, points_list):
    task_description = data_line['text']
    video_path = data_line["video_path"]

    # 提取帧
    frames = extract_frames(video_path, frame_inds=points_list)

    # 生成 JSONL 条目
    jsonl_entry = generate_jsonl_entry(video_path, frames, task_description)
    return jsonl_entry

def vid_frame_extract_and_create_batch_request(data, input_dir, output_dir, jsonl_output_path, points, max_workers=32):
    num_points = len(points)
    data = pd.DataFrame(np.repeat(data.values, num_points, axis=0), columns=data.columns)
    data["point"] = np.nan
    data["video_path"] = data["path"]  # Add the original video path to a new column

    for i, point in enumerate(points):
        if isinstance(point, int):
            data.loc[i::num_points, "point"] = point
        else:
            data.loc[i::num_points, "point"] = data.loc[i::num_points, "num_frames"] * point

    jsonl_entries = []

    # Group by video path to ensure frames from the same video are grouped together
    grouped_data = data.groupby('path').agg({
        'point': list,
        'text': 'first',
        'num_frames': 'first',
        'height': 'first',
        'width': 'first',
        'aspect_ratio': 'first',
        'fps': 'first',
        'resolution': 'first',
        'video_path': 'first'
    }).reset_index()

    # 使用 ThreadPoolExecutor 进行多线程处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for data_line in tqdm(grouped_data.iterrows(), total=len(grouped_data), desc="Processing videos"):
            data_line = data_line[1]  # 获取数据行
            points_list = data_line["point"]
            futures.append(executor.submit(process_video, data_line, output_dir, points_list))

        # 收集多线程执行结果
        batch_size = 100  # 设置批量大小
        batch_entries = []
        for future in tqdm(as_completed(futures), total=len(futures), desc="Collecting results"):
            try:
                jsonl_entry = future.result()
                if jsonl_entry:  # 确保没有重复帧的条目才保存
                    batch_entries.append(jsonl_entry)
                
                # 批量写入文件
                if len(batch_entries) >= batch_size:
                    with open(jsonl_output_path, 'a') as f:
                        for entry in batch_entries:
                            f.write(json.dumps(entry) + "\n")
                    batch_entries = []  # 清空批处理条目

            except Exception as e:
                print(f"Error processing video: {e}")
        
        # 写入剩余的条目
        if batch_entries:
            with open(jsonl_output_path, 'a') as f:
                for entry in batch_entries:
                    f.write(json.dumps(entry) + "\n")

    print(f"Batch requests saved to {jsonl_output_path}")




def main(args):
    data = pd.read_csv(args.input)
    if args.method == "img_rand_crop":
        data["path"] = apply(data["path"], lambda x: rand_crop(x, args.input_dir, args.output))
        output_csv = args.input.replace(".csv", f"_rand_crop.csv")
    elif args.method == "img_resize":
        data["path"] = apply(data["path"], lambda x: resize(x, args.length, args.input_dir, args.output))
        output_csv = args.input.replace(".csv", f"_resized{args.length}.csv")
    elif args.method == "vid_frame_extract_and_create_batch_request":
        points = args.points if args.points is not None else args.points_index
        vid_frame_extract_and_create_batch_request(data, args.input_dir, args.output, args.jsonl_output, points)
        output_csv = args.input.replace(".csv", f"_vid_frame_extract.csv")

    data.to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", type=str, choices=["img_resize", "img_rand_crop", "vid_frame_extract_and_create_batch_request"])
    parser.add_argument("input", type=str)
    parser.add_argument("input_dir", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("--jsonl_output", type=str, required=True, help="Path to save the JSONL batch requests")
    parser.add_argument("--disable-parallel", action="store_true")
    parser.add_argument("--length", type=int, default=2160)
    parser.add_argument("--seed", type=int, default=42, help="seed for random")
    parser.add_argument("--points", nargs="+", type=float, default=None)
    parser.add_argument("--points_index", nargs="+", type=int, default=None)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    if args.disable_parallel:
        pandas_has_parallel = False
    main(args)
