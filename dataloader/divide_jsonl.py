import json
import os

def split_jsonl_file(input_file, output_dir, max_lines_per_file=50000, max_size_mb=100):
    os.makedirs(output_dir, exist_ok=True)
    
    max_size_bytes = max_size_mb * 1024 * 1024  # Convert MB to bytes
    file_counter = 1
    line_counter = 0
    current_file_size = 0
    
    output_file = open(os.path.join(output_dir, f'part_{file_counter}.jsonl'), 'w')
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line_size = len(line.encode('utf-8'))
            
            # Check if we need to start a new file
            if line_counter >= max_lines_per_file or (current_file_size + line_size) > max_size_bytes:
                output_file.close()
                file_counter += 1
                line_counter = 0
                current_file_size = 0
                output_file = open(os.path.join(output_dir, f'part_{file_counter}.jsonl'), 'w')
            
            # Write the line to the current file
            output_file.write(line)
            line_counter += 1
            current_file_size += line_size

    output_file.close()

# 使用示例
input_file = 'data/caption/batch_request_droid.jsonl'  # 原始JSONL文件路径
output_dir = 'data/caption/requests'  # 输出文件夹路径
split_jsonl_file(input_file, output_dir)
