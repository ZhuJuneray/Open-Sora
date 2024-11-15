import json
import csv
import os
from pathlib import Path
import warnings

def process_json_files(input_folder, output_csv):
    """
    递归处理文件夹中的所有JSON文件并将数据写入CSV
    
    Args:
        input_folder: 输入文件夹路径
        output_csv: 输出CSV文件路径
    """
    # 创建CSV文件并写入表头
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['path', 'text'])
        
        # 递归遍历所有JSON文件
        for json_file in Path(input_folder).rglob('*.json'):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 获取并组合所有文本
                texts = ' '.join(data.get('texts', []))
                
                # 处理视频路径
                videos = data.get('videos', [])
                
                # 如果有多个视频，发出警告
                if len(videos) > 1:
                    warnings.warn(f"文件 {json_file} 包含多个视频路径")
                
                # 为每个视频路径创建一行
                for video in videos:
                    video_path = video.get('video_path', '')
                    csv_writer.writerow(['dataset/bridge/'+video_path, texts])
                    
            except Exception as e:
                print(f"处理文件 {json_file} 时出错: {str(e)}")

# 使用示例
if __name__ == "__main__":
    input_folder = "dataset/opensource_robotdata/bridge/annotation"  # 替换为你的输入文件夹路径
    output_csv = "dataloader/bridge/bridge.csv"  # 输出CSV文件名
    
    process_json_files(input_folder, output_csv)
    print(f"转换完成，结果已保存至 {output_csv}")