import os
import random
import shutil

# 设置随机种子以便重现
random.seed(42)

# 定义文件路径
source_folder = '/shared/fangchen/junrui/Open-Sora/dataset/opensource_robotdata/droid/annotation'  # 原始json文件所在的文件夹路径
train_folder = '/shared/fangchen/junrui/Open-Sora/dataset/opensource_robotdata/droid_sample/annotation/train'      # 训练集文件夹路径
val_folder = '/shared/fangchen/junrui/Open-Sora/dataset/opensource_robotdata/droid_sample/annotation/val'   # 验证集文件夹路径
test_folder = '/shared/fangchen/junrui/Open-Sora/dataset/opensource_robotdata/droid_sample/annotation/test'        # 测试集文件夹路径

# 创建目标文件夹（如果不存在）
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# 获取所有json文件
files = [f for f in os.listdir(source_folder) if f.endswith('.json')]

# 随机打乱文件列表
random.shuffle(files)

# 定义数据集划分比例
train_ratio = 0.07
val_ratio = 0.015
test_ratio = 0.015

# 计算每个数据集的大小
total_files = len(files)
train_size = int(total_files * train_ratio)
val_size = int(total_files * val_ratio)
test_size = int(total_files * test_ratio)

# 分配文件到不同的数据集
train_files = files[:train_size]
val_files = files[train_size:train_size + val_size]
test_files = files[train_size + val_size:]

# 移动文件到相应的文件夹
for f in train_files:
    shutil.move(os.path.join(source_folder, f), os.path.join(train_folder, f))

for f in val_files:
    shutil.move(os.path.join(source_folder, f), os.path.join(val_folder, f))

for f in test_files:
    shutil.move(os.path.join(source_folder, f), os.path.join(test_folder, f))

print(f'Dataset split completed: {train_size} train, {val_size} validation, {test_size} test files.')
