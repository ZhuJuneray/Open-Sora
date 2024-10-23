# Training on DROID Pipeline

## Data Process
### suppose videos are under ../droid-raw/droid_raw/
1. Convert dataset to CSV, output is column "path" and "text" in CSV
```bash
python -m tools.datasets.convert droid /shared/fangchen/junrui/Open-Sora/dataset/droid_480p24fps --output dataloader/droid_480p24fps.csv
```
2. Get video information, output is column "num_frames,height,width,aspect_ratio,fps,resolution" in CSV
```bash
python -m tools.datasets.datautil dataloader/droid_480p24fps.csv --video-info
```

## Model config
### suppose at configs/opensora-v1-2/train/droid.py
#### Need to reset bucket_config
#### To use bucket, first search batch size for buckets:
```bash
torchrun --standalone --nproc_per_node 1 scripts/misc/search_bs.py configs/opensora-v1-2/misc/bs.py --data-path /shared/fangchen/junrui/Open-Sora/dataset_csv/droid_vinfo_bs.csv
```
#### detailed in docs/commands.md

### Also some other hyperparameters need to be set
#### detalied in docs/cofig.md

## Training
```bash
# one node
CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node 4 scripts/train.py configs/opensora-v1-2/train/droid.py --data-path dataloader/droid_docker.csv

# multiple nodes
colossalai run --nproc_per_node 8 --hostfile hostfile scripts/train.py \
    configs/opensora-v1-2/train/droid.py --data-path dataset_csv/droid_vinfo.csv --ckpt-path YOUR_PRETRAINED_CKPT
```


# Using Docker
remember to specify ipconfig
```bash
# install
docker build --network=host -t opensora .
# run
sudo docker run -ti --gpus all -v /home/fangchen/junrui/Open-Sora:/workspace/Open-Sora -v /shared/fangchen/junrui/Open-Sora/dataset/droid_raw.:/workspace/Open-Sora/dataset --network host opensora # 挂载不上不懂为什么

sudo docker run -ti --gpus all -v /shared/fangchen/junrui/Open-Sora/dataset:/workspace/dataset -v .:/workspace/Open-Sora --network host --ipc=host opensora # 用这个

sudo docker run -ti --gpus all \
    --user $(id -u):$(id -g) \
    -v /shared/fangchen/junrui/Open-Sora/dataset:/workspace/dataset \
    -v $(pwd):/workspace/Open-Sora \
    --network host \
    --ipc=host \
    opensora

docker run -ti --gpus all --mount type=bind,source=/shared/fangchen/junrui/Open-Sora/dataset,target=/workspace/dataset,bind-propagation=shared --network host --ipc=host opensora

sudo docker run -ti --gpus all -v .:/workspace/opensora --network host opensora
sudo docker run -ti --gpus all --network host opensora
sudo docker run -ti --gpus all -v .:/workspace/Open-Sora/test_folder --network host opensora # target path要选在pwd下的某个文件或文件夹，就能挂载
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node 4 scripts/train.py configs/opensora-v1-2/train/droid.py --data-path dataloader/droid_docker.csv

```

# Sample
```bash
# text to video
CUDA_VISIBLE_DEVICES=6,7 OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node 2 scripts/inference.py configs/opensora-v1-2/inference/sample.py \
  --num-frames 4s --resolution 240p --aspect-ratio 9:16 \
  --prompt "A robot is folding a cloth."
```

# Frame Extraction and generate batch request
```bash
python -m tools.datasets.transform vid_frame_extract_and_create_batch_request dataloader/droid.csv /shared/fangchen/junrui/Open-Sora/dataset/droid_480p24fps /shared/fangchen/junrui/Open-Sora/dataset/droid_480p24fps_frame --jsonl_output "/home/fangchen/junrui/Open-Sora/data/caption/batch_request_droid.jsonl" --points 0.1 0.3 0.6 0.9
```

# filter out very short videos
```bash
python -m tools.datasets.datautil /home/fangchen/junrui/Open-Sora/dataloader/droid_480p24fps_vinfo.csv --min_frame 12 --output /home/fangchen/junrui/Open-Sora/dataloader/droid.csv
```
droid.csv has been filtered out
droid_caption.csv is caption augmented

# wormhole send file
package: 
tar cf - /home/fangchen/junrui/Open-Sora/dataset/droid_480p24fps | pigz -p 32 > /shared/fangchen/junrui/Open-Sora/dataset/droid_480p24fps.tar.gz
send: 
wormhole send dataset/droid_480p24fps.tar.gz

# Train VAE
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=4 scripts/train_vae.py configs/vae/train/stage3.py --data-path dataloader/droid_docker.csv
```

# Test VAE
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=4 scripts/inference_vae.py configs/vae/inference/video.py --ckpt-path hpcai-tech/OpenSora-VAE-v1.2 --data-path dataloader/test_sample.csv --save-dir samples/vae_test_start
```

# Inference with checkpoints saved during training
torchrun --standalone --nproc_per_node 2 scripts/inference.py configs/opensora/inference/16x256x256.py --ckpt-path outputs/001-STDiT-XL-2/epoch12-global_step2000