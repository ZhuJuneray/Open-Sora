import pandas as pd

def merge_csv_files(file1_path, file2_path, output_path, sample_output_path):
    # 读取两个CSV文件
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)
    
    # 使用path列作为索引进行合并，使用outer join获取并集
    merged_df = pd.merge(df1, df2, on='path', how='outer')
    
    # 保存完整的合并结果
    merged_df.to_csv(output_path, index=False)
    
    # 保存前三行作为样本
    merged_df.head(3).to_csv(sample_output_path, index=False)
    
    # 打印一些基本信息
    print(f"原始文件1的行数: {len(df1)}")
    print(f"原始文件2的行数: {len(df2)}")
    print(f"合并后的行数: {len(merged_df)}")
    print(f"合并后的列: {list(merged_df.columns)}")

# 使用示例
file1_path = 'dataloader/bridge/bridge.csv'
file2_path = 'dataloader/bridge/bridge_actioned.csv'
output_path = 'dataloader/bridge/bridge_action.csv'
sample_output_path = 'dataloader/bridge/bridge_action_sample.csv'

merge_csv_files(file1_path, file2_path, output_path, sample_output_path)