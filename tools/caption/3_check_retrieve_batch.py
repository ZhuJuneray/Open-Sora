# retieve a dir
from openai import OpenAI
import json
import os

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Specify the directory containing the info JSON files
info_directory = "data/caption/info"
output_directory = "data/caption/response"

# Iterate over each file in the info directory
for filename in os.listdir(info_directory):
    if filename.endswith("_info.json"):
        # Read the info JSON file
        info_file_path = os.path.join(info_directory, filename)
        with open(info_file_path, "r") as json_file:
            response_data = json.load(json_file)
        
        # Retrieve the batch using the ID from the info file
        batch = client.batches.retrieve(response_data["id"])
        # print(f"Batch Status for {filename}: {batch.status}")
        
        # Define output file names
        completed_info_filename = filename.replace("_info.json", "_completed_info.json")
        output_filename = filename.replace("_info.json", "_response.jsonl")
        
        # Check if batch is completed
        if batch.status == "completed":
            # Skip if the output file already exists
            if os.path.exists(os.path.join(output_directory, output_filename)):
                # print(f"Skipping {filename}: Output file already exists.")
                continue
            
            # Save batch info response to a file
            with open(os.path.join(output_directory, completed_info_filename), "w") as json_file:
                json.dump(batch.to_dict(), json_file, indent=4)
            # print(batch)

            # Retrieve the output file content
            file_response = client.files.content(batch.output_file_id)
            file_content = file_response.read()
            
            # Save the output content to a new JSONL file
            with open(os.path.join(output_directory, output_filename), "wb") as json_file:
                json_file.write(file_content)

        else:
            # Print status if it's not completed
            print(f"Skipping {filename}: Status is '{batch.status}'.")

print("Script end.")

# single file: 

# from openai import OpenAI
# import json
# client = OpenAI()
# with open("/home/fangchen/junrui/Open-Sora/data/caption/batch_request_droid_480p24fps_info.json", "r") as json_file:
#     response_data = json.load(json_file)
# batch = client.batches.retrieve(response_data["id"])
# print(batch.status)
# if batch.status == "completed":
#     # save batch_info response to a file
#     with open("/home/fangchen/junrui/Open-Sora/data/caption/batch_request_droid_480p24fps_completed_info.json", "w") as json_file:
#         json.dump(batch.to_dict(), json_file, indent=4)
#     print(batch)
#     file_response = client.files.content(batch.output_file_id)
#     file_content = file_response.read()
#     with open("/home/fangchen/junrui/Open-Sora/data/caption/batch_response_droid_480p24fps.jsonl", "wb") as json_file:
#         json_file.write(file_content)


