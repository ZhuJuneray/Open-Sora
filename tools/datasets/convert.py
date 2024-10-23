import argparse
import os
import time
import json

import pandas as pd
from torchvision.datasets import ImageNet

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")
VID_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".m2ts")


def scan_recursively(root):
    num = 0
    for entry in os.scandir(root):
        if entry.is_file():
            yield entry
        elif entry.is_dir():
            num += 1
            if num % 100 == 0:
                print(f"Scanned {num} directories.")
            yield from scan_recursively(entry.path)


def get_filelist(file_path, exts=None):
    filelist = []
    time_start = time.time()

    # == OS Walk ==
    # for home, dirs, files in os.walk(file_path):
    #     for filename in files:
    #         ext = os.path.splitext(filename)[-1].lower()
    #         if exts is None or ext in exts:
    #             filelist.append(os.path.join(home, filename))

    # == Scandir ==
    obj = scan_recursively(file_path)
    for entry in obj:
        if entry.is_file():
            ext = os.path.splitext(entry.name)[-1].lower()
            if exts is None or ext in exts:
                filelist.append(entry.path)

    time_end = time.time()
    print(f"Scanned {len(filelist)} files in {time_end - time_start:.2f} seconds.")
    return filelist


def split_by_capital(name):
    # BoxingPunchingBag -> Boxing Punching Bag
    new_name = ""
    for i in range(len(name)):
        if name[i].isupper() and i != 0:
            new_name += " "
        new_name += name[i]
    return new_name


def process_imagenet(root, split):
    root = os.path.expanduser(root)
    data = ImageNet(root, split=split)
    samples = [(path, data.classes[label][0]) for path, label in data.samples]
    output = f"imagenet_{split}.csv"

    df = pd.DataFrame(samples, columns=["path", "text"])
    df.to_csv(output, index=False)
    print(f"Saved {len(samples)} samples to {output}.")


def process_ucf101(root, split):
    root = os.path.expanduser(root)
    video_lists = get_filelist(os.path.join(root, split))
    classes = [x.split("/")[-2] for x in video_lists]
    classes = [split_by_capital(x) for x in classes]
    samples = list(zip(video_lists, classes))
    output = f"ucf101_{split}.csv"

    df = pd.DataFrame(samples, columns=["path", "text"])
    df.to_csv(output, index=False)
    print(f"Saved {len(samples)} samples to {output}.")


def process_vidprom(root, info):
    root = os.path.expanduser(root)
    video_lists = get_filelist(root)
    video_set = set(video_lists)
    # read info csv
    infos = pd.read_csv(info)
    abs_path = infos["uuid"].apply(lambda x: os.path.join(root, f"pika-{x}.mp4"))
    is_exist = abs_path.apply(lambda x: x in video_set)
    df = pd.DataFrame(dict(path=abs_path[is_exist], text=infos["prompt"][is_exist]))
    df.to_csv("vidprom.csv", index=False)
    print(f"Saved {len(df)} samples to vidprom.csv.")


def process_general_images(root, output):
    root = os.path.expanduser(root)
    if not os.path.exists(root):
        return
    path_list = get_filelist(root, IMG_EXTENSIONS)
    fname_list = [os.path.splitext(os.path.basename(x))[0] for x in path_list]
    df = pd.DataFrame(dict(id=fname_list, path=path_list))

    os.makedirs(os.path.dirname(output), exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Saved {len(df)} samples to {output}.")


def process_general_videos(root, output):
    root = os.path.expanduser(root)
    if not os.path.exists(root):
        return
    path_list = get_filelist(root, VID_EXTENSIONS)
    path_list = list(set(path_list))  # remove duplicates
    fname_list = [os.path.splitext(os.path.basename(x))[0] for x in path_list]
    relpath_list = [os.path.relpath(x, root) for x in path_list]
    df = pd.DataFrame(dict(path=path_list, id=fname_list, relpath=relpath_list))

    os.makedirs(os.path.dirname(output), exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Saved {len(df)} samples to {output}.")

def process_droid(root, output):
    root = os.path.expanduser(root)
    video_list = []
    task_list = []

    # Recursively scan for mp4 files
    for subdir, _, files in os.walk(root):
        mp4_files = [f for f in files if f.endswith('.mp4')]
        
        # For each mp4 file, find the corresponding JSON file two levels up
        if mp4_files:
            # Get the parent directory and look for the JSON file
            parent_dir = os.path.dirname(os.path.dirname(subdir))  # Two levels up
            json_file = None

            # Find the JSON file starting with 'metadata'
            for file in os.listdir(parent_dir):
                if file.startswith("metadata") and file.endswith(".json"):
                    json_file = os.path.join(parent_dir, file)
                    break

            # If the JSON file exists, read it and get the "current_task" field
            if json_file and os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    metadata = json.load(f)
                    current_task = metadata.get("current_task", "Unknown Task")
                    
                # Add each video file with its corresponding task
                for mp4 in mp4_files:
                    video_path = os.path.join(subdir, mp4)
                    video_list.append(video_path)
                    task_list.append(current_task)

    # Create DataFrame and save to CSV
    df = pd.DataFrame({"path": video_list, "text": task_list})
    os.makedirs(os.path.dirname(output), exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Saved {len(df)} samples to {output}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, choices=["imagenet", "ucf101", "vidprom", "droid", "image", "video"])
    parser.add_argument("root", type=str)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--info", type=str, default=None)
    parser.add_argument("--output", type=str, default=None, required=True, help="Output path")
    args = parser.parse_args()

    if args.dataset == "imagenet":
        process_imagenet(args.root, args.split)
    elif args.dataset == "ucf101":
        process_ucf101(args.root, args.split)
    elif args.dataset == "vidprom":
        process_vidprom(args.root, args.info)
    elif args.dataset == "image":
        process_general_images(args.root, args.output)
    elif args.dataset == "video":
        process_general_videos(args.root, args.output)
    elif args.dataset == "droid":
        process_droid(args.root, args.output)
    else:
        raise ValueError("Invalid dataset")
