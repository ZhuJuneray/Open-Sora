import pandas as pd

# 读取CSV文件
df = pd.read_csv('dataset_csv/droid_vinfo.csv')

# 替换'path'列中的字符串
df['path'] = df['path'].str.replace('../droid-raw/', './data')

# 将修改后的数据写入新的CSV文件
df.to_csv('dataset_csv/droid_docker.csv', index=False)