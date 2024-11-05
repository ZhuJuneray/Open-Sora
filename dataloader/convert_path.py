import pandas as pd

# 读取CSV文件
df = pd.read_csv('dataloader/full_dataset/droid_raw_vinfo.csv')

# 替换'path'列中的字符串
df['path'] = df['path'].str.replace('../droid-raw/droid_raw', 'dataset/droid_raw')

# 将修改后的数据写入新的CSV文件
df.to_csv('dataloader/full_dataset/droid_full_vinfo.csv', index=False)