# Training on DROID Pipeline

## Data Process
### suppose videos are under ../droid-raw/droid_raw/
1. Convert dataset to CSV, output is column "path" and "text" in CSV
```bash
python -m tools.datasets.convert droid ../droid-raw/droid_raw --output dataset_csv/droid.csv
```
2. Get video information, output is column "num_frames,height,width,aspect_ratio,fps,resolution" in CSV
```bash
python -m tools.datasets.datautil dataset_csv/droid.csv --video-info
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
CUDA_VISIBLE_DEVICES=3 torchrun --standalone --nproc_per_node 1 scripts/train.py \
    configs/opensora-v1-2/train/droid.py --data-path dataset_csv/droid_vinfo.csv --ckpt-path YOUR_PRETRAINED_CKPT
# multiple nodes
colossalai run --nproc_per_node 8 --hostfile hostfile scripts/train.py \
    configs/opensora-v1-2/train/droid.py --data-path dataset_csv/droid_vinfo.csv --ckpt-path YOUR_PRETRAINED_CKPT
```


# Using Docker
remember to specify ipconfig
```bash
docker build --network=host -t opensora .
sudo docker run -ti --gpus all -v .:/workspace/Open-Sora --network host opensora # 挂载不上不懂为什么
sudo docker run -ti --gpus all -v .:/workspace/opensora --network host opensora
sudo docker run -ti --gpus all --network host opensora
sudo docker run -ti --gpus all -v .:/workspace/Open-Sora/test_folder --network host opensora # target path要选在pwd下的某个文件或文件夹，就能挂载
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node 1 scripts/train.py \
    configs/opensora-v1-2/train/droid.py --data-path dataset_csv/droid_docker.csv

```