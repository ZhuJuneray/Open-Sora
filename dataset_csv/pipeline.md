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
torchrun --standalone --nproc_per_node 1 scripts/misc/search_bs.py configs/opensora-v1-2/misc/bs.py --data-path /mnt/nfs-207/sora_data/meta/searchbs.csv
```
#### detailed in docs/commands.md

### Also some other hyperparameters need to be set
#### detalied in docs/cofig.md

## Training
```bash
# one node
torchrun --standalone --nproc_per_node 8 scripts/train.py \
    configs/opensora-v1-2/train/droid.py --data-path dataset_csv/droid_vinfo.csv --ckpt-path YOUR_PRETRAINED_CKPT
# multiple nodes
colossalai run --nproc_per_node 8 --hostfile hostfile scripts/train.py \
    configs/opensora-v1-2/train/droid.py --data-path dataset_csv/droid_vinfo.csv --ckpt-path YOUR_PRETRAINED_CKPT
```