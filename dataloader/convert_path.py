import pandas as pd

# 读取CSV文件
df = pd.read_csv('dataloader/droid_caption.csv')

# 替换'path'列中的字符串
df['path'] = df['path'].str.replace('/shared/fangchen/junrui/Open-Sora/dataset/droid_480p24fps', 'dataset/droid_480p24fps')

# 将修改后的数据写入新的CSV文件
df.to_csv('dataloader/droid_docker.csv', index=False)