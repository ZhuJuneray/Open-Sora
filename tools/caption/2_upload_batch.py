# from openai import OpenAI
# import json
# import os

# client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# # 上传文件，创建 batch 输入文件
# batch_input_file = client.files.create(
#   file=open("/home/fangchen/junrui/Open-Sora/data/caption/batch_request_droid_480p24fps.jsonl", "rb"),
#   purpose="batch"
# )

# # 获取文件ID
# batch_input_file_id = batch_input_file.id

# # 创建 batch 请求
# response = client.batches.create(
#     input_file_id=batch_input_file_id,
#     endpoint="/v1/chat/completions",
#     completion_window="24h",
#     metadata={
#       "description": "to caption videos",
#     }
# )

# # 检查是否有方法将 response 转换为字典
# if hasattr(response, 'to_dict'):
#     response_data = response.to_dict()
# else:
#     # 如果没有 to_dict 方法，可以手动从 response 中提取有用的信息
#     response_data = {
#         "id": response.id,
#         "status": response.status,
#         "created_at": response.created_at,
#         "metadata": response.metadata,
#         # 根据需要添加其他属性
#     }

# # 将提取的数据写入json文件
# with open("/home/fangchen/junrui/Open-Sora/data/caption/batch_request_droid_480p24fps_info.json", "w") as json_file:
#     json.dump(response_data, json_file, indent=4)

# multiple batch
from openai import OpenAI
import json
import os

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Specify the directory containing the JSONL files
input_directory = "/home/fangchen/junrui/Open-Sora/data/caption/droid/request"
output_directory = "/home/fangchen/junrui/Open-Sora/data/caption/droid/info"

# Iterate over each file in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith(".jsonl"):
        # Upload file to create batch input file
        file_path = os.path.join(input_directory, filename)
        batch_input_file = client.files.create(
            file=open(file_path, "rb"),
            purpose="batch"
        )

        # Get the file ID
        batch_input_file_id = batch_input_file.id

        # Create batch request
        response = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": f"to caption videos from {filename}",
            }
        )

        # Check if response can be converted to a dictionary
        if hasattr(response, 'to_dict'):
            response_data = response.to_dict()
        else:
            # Manually extract useful information
            response_data = {
                "id": response.id,
                "status": response.status,
                "created_at": response.created_at,
                "metadata": response.metadata,
            }

        # Write response data to a JSON file
        info_filename = filename.replace(".jsonl", "_info.json")
        with open(os.path.join(output_directory, info_filename), "w") as json_file:
            json.dump(response_data, json_file, indent=4)