import os
import json
import csv
from pathlib import Path

def process_json_files(root_dir, output_csv):
    """
    遍历文件夹中的所有JSON文件，提取特定字段并生成CSV文件
    
    Args:
        root_dir (str): 要遍历的根目录路径
        output_csv (str): 输出CSV文件的路径
    """
    # 用于存储所有处理后的路径
    processed_paths = []
    
    # 遍历所有子文件夹
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.json'):
                json_path = os.path.join(root, file)
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    # 提取需要的字段
                    if 'right_mp4_path' in data and 'lab' in data:
                        combined_path = os.path.join(
                            "dataset/droid_480p24fps",
                            data['lab'],
                            data['right_mp4_path']
                        ).replace('\\', '/')  # 确保使用正斜杠
                        
                        processed_paths.append([combined_path])
                        
                except json.JSONDecodeError as e:
                    print(f"Error reading JSON file {json_path}: {e}")
                except Exception as e:
                    print(f"Unexpected error processing {json_path}: {e}")
    
    # 将结果写入CSV文件
    try:
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['path'])  # 写入表头
            writer.writerows(processed_paths)
        print(f"Successfully processed {len(processed_paths)} files")
        print(f"Results saved to {output_csv}")
    except Exception as e:
        print(f"Error writing CSV file: {e}")

# 使用示例
if __name__ == "__main__":
    # 替换为你的实际路径
    root_directory = "dataset/droid_480p24fps"
    output_csv_path = "dataloader/droid/480p24fps/droid_right.csv"
    
    process_json_files(root_directory, output_csv_path)