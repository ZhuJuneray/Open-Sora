# Training on DROID Pipeline

## Data Process
Convert dataset to CSV, output is column "path" and "text" in CSV
```bash
python -m tools.datasets.convert droid /shared/fangchen/junrui/Open-Sora/dataset/droid_480p3fps --output dataloader/480p3fps/droid_480p3fps.csv
```
Get video information, output is column "num_frames,height,width,aspect_ratio,fps,resolution" in CSV
```bash
python -m tools.datasets.datautil dataloader/full_dataset/droid_raw.csv --video-info
```

## Training
```bash
# one node
CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node 4 scripts/train.py configs/opensora-v1-2/train/droid.py --data-path dataloader/droid_docker.csv

# multiple nodes
colossalai run --nproc_per_node 8 --hostfile hostfile scripts/train.py \
    configs/opensora-v1-2/train/droid.py --data-path dataset_csv/droid_vinfo.csv --ckpt-path YOUR_PRETRAINED_CKPT
```


# Inference Sample
```bash
# text to video
CUDA_VISIBLE_DEVICES=6,7 OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node 2 scripts/inference.py configs/opensora-v1-2/inference/sample.py \
  --num-frames 4s --resolution 240p --aspect-ratio 9:16 \
  --prompt "A robot is folding a cloth."
```

# Use ChatGPT to generate video caption
## Frame Extraction and generate batch request
```bash
python -m tools.caption.batch_request \
    dataloader/droid/480p24fps/droid_nocaption_sample.csv \ # dataset csv
    dataset/droid_480p24fps \ # video dataset path
    dataset/droid_480p24fps_frame \ # those failed to extract will be save here (num_frame=1)
    --jsonl_output "data/caption/batch_request_droid.jsonl" \
    --template_path "tools/caption/templates/message_template.txt" \ # prompt template
    --points 0.1 0.3 0.6 0.9 # which frames to be extracted
```
Now we have generated jsonl file "data/caption/batch_request_droid.jsonl" to be uploaded to OpenAI
If the jsonl file is oversized, run dataloader/divide_jsonl.py to divide it into chunks.
## Upload jsonl batch file
run tools/caption/2_upload_batch.py
Remember to check variables
## Check status and retrieve response (if completed)
run tools/caption/3_check_retrieve_batch.py


# filter out very short videos
```bash
python -m tools.datasets.datautil /home/fangchen/junrui/Open-Sora/dataloader/droid_480p24fps_vinfo.csv --min_frame 12 --output /home/fangchen/junrui/Open-Sora/dataloader/droid.csv
```

# wormhole send file
package in parallel: 
tar cf - /home/fangchen/junrui/Open-Sora/dataset/droid_480p24fps | pigz -p 32 > /shared/fangchen/junrui/Open-Sora/dataset/droid_480p24fps.tar.gz
send: 
wormhole send dataset/droid_480p24fps.tar.gz

# VAE
## Train VAE
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=4 scripts/train_vae.py configs/vae/train/stage3.py --data-path dataloader/droid_docker.csv
```

## Test VAE
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=4 scripts/inference_vae.py configs/vae/inference/video.py --ckpt-path hpcai-tech/OpenSora-VAE-v1.2 --data-path dataloader/test_sample.csv --save-dir samples/vae_test_start
```

pretrained VAE is good
