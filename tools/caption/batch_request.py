import argparse
import os
import random
import base64
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from tqdm import tqdm

import numpy as np
import pandas as pd
from PIL import Image

# 从datasets导入必要的工具函数
from ..datasets.utils import extract_frames

tqdm.pandas()

try:
    from pandarallel import pandarallel
    pandarallel.initialize(progress_bar=True)
    pandas_has_parallel = True
except ImportError:
    pandas_has_parallel = False

def load_message_template(template_path):
    """Load message template from a file."""
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"Template file not found at {template_path}")
    except Exception as e:
        raise Exception(f"Error reading template file: {e}")

def image_to_base64(image):
    """Convert PIL Image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def generate_jsonl_entry(video_path, frames, task_description, template_path):
    """Generate a JSONL entry for a video with its frames and task description."""
    # Encode frames to base64
    base64_images = [image_to_base64(frame) for frame in frames]
    
    # Check for duplicate frames
    if len(set(base64_images)) != len(frames):
        print(f"Warning: Duplicate frames detected in {video_path}")
        return None
    
    # Load and format template
    try:
        message_template = load_message_template(template_path)
        message_content = message_template.format(task_description=task_description)
    except Exception as e:
        print(f"Error loading template for {video_path}: {e}")
        return None
    
    # Create image entries
    image_entries = [
        {"type": "image_url", 
         "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}} 
        for base64_image in base64_images
    ]
    
    # Create JSONL entry
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
                        {"type": "text", "text": message_content},
                        *image_entries
                    ]
                }
            ],
            "temperature": 0,
            "max_tokens": 1000
        }
    }
    
    return jsonl_entry

def process_video(data_line, output_dir, points_list, template_path):
    """Process a single video: extract frames and generate JSONL entry."""
    task_description = data_line['text']
    video_path = data_line["video_path"]
    
    # Extract frames
    frames = extract_frames(video_path, frame_inds=points_list)
    
    # Generate JSONL entry
    return generate_jsonl_entry(video_path, frames, task_description, template_path)

def create_batch_request(data, input_dir, output_dir, jsonl_output_path, points, template_path, max_workers=128):
    """Create batch request for multiple videos."""
    # 复制并展开数据帧
    num_points = len(points)
    data = pd.DataFrame(np.repeat(data.values, num_points, axis=0), columns=data.columns)
    data["point"] = np.nan
    data["video_path"] = data["path"]
    
    # 设置采样点
    for i, point in enumerate(points):
        if isinstance(point, int):
            data.loc[i::num_points, "point"] = point
        else:
            data.loc[i::num_points, "point"] = data.loc[i::num_points, "num_frames"] * point
    
    # 按视频路径分组
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
    
    # 多线程处理视频
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for _, data_line in tqdm(grouped_data.iterrows(), total=len(grouped_data), desc="Processing videos"):
            points_list = data_line["point"]
            futures.append(
                executor.submit(process_video, data_line, output_dir, points_list, template_path)
            )
        
        # 批量写入结果
        batch_size = 100
        batch_entries = []
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Collecting results"):
            try:
                jsonl_entry = future.result()
                if jsonl_entry:  # 只保存非重复帧的条目
                    batch_entries.append(jsonl_entry)
                    
                    # 达到批量大小时写入文件
                    if len(batch_entries) >= batch_size:
                        with open(jsonl_output_path, 'a') as f:
                            for entry in batch_entries:
                                f.write(json.dumps(entry) + "\n")
                        batch_entries = []
                        
            except Exception as e:
                print(f"Error processing video: {e}")
        
        # 写入剩余条目
        if batch_entries:
            with open(jsonl_output_path, 'a') as f:
                for entry in batch_entries:
                    f.write(json.dumps(entry) + "\n")
    
    print(f"Batch requests saved to {jsonl_output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input CSV file")
    parser.add_argument("input_dir", type=str, help="Input directory containing videos")
    parser.add_argument("output_dir", type=str, help="Output directory")
    parser.add_argument("--jsonl_output", type=str, required=True, 
                       help="Path to save the JSONL batch requests")
    parser.add_argument("--template_path", type=str, required=True,
                       help="Path to the message template file")
    parser.add_argument("--disable-parallel", action="store_true")
    parser.add_argument("--points", nargs="+", type=float, default=None)
    parser.add_argument("--points_index", nargs="+", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-workers", type=int, default=32,
                       help="Maximum number of worker threads")
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 设置并行处理
    if args.disable_parallel:
        global pandas_has_parallel
        pandas_has_parallel = False
    
    # 读取数据
    data = pd.read_csv(args.input)
    points = args.points if args.points is not None else args.points_index
    
    # 创建batch请求
    create_batch_request(
        data=data,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        jsonl_output_path=args.jsonl_output,
        points=points,
        template_path=args.template_path,
        max_workers=args.max_workers
    )

if __name__ == "__main__":
    main()