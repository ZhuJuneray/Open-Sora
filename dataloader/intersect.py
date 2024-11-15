import pandas as pd

def filter_csv_by_path(complete_csv_path, filter_csv_path, output_csv_path):
    """
    使用只包含path的CSV文件作为筛选条件，从完整的CSV文件中筛选出对应的行
    
    Args:
        complete_csv_path (str): 完整CSV文件的路径（包含所有字段）
        filter_csv_path (str): 用于筛选的CSV文件路径（只包含path字段）
        output_csv_path (str): 输出CSV文件的路径
    """
    try:
        # 读取两个CSV文件
        complete_df = pd.read_csv(complete_csv_path)
        filter_df = pd.read_csv(filter_csv_path)
        
        # 确保两个DataFrame都有path列
        if 'path' not in complete_df.columns or 'path' not in filter_df.columns:
            raise ValueError("Both CSV files must contain a 'path' column")
            
        # 获取filter_df中的所有path值
        filter_paths = set(filter_df['path'].values)
        
        # 使用isin方法筛选完整DataFrame中的行
        filtered_df = complete_df[complete_df['path'].isin(filter_paths)]
        
        # 计算统计信息
        total_filter_paths = len(filter_paths)
        total_complete_paths = len(complete_df)
        matched_paths = len(filtered_df)
        unmatched_paths = total_filter_paths - matched_paths
        
        # 保存结果
        filtered_df.to_csv(output_csv_path, index=False)
        
        # 打印统计信息
        print(f"处理完成！统计信息如下：")
        print(f"筛选文件中的路径总数: {total_filter_paths}")
        print(f"完整文件中的路径总数: {total_complete_paths}")
        print(f"成功匹配的路径数: {matched_paths}")
        print(f"未能匹配的路径数: {unmatched_paths}")
        print(f"结果已保存至: {output_csv_path}")
        
        # 如果有未匹配的路径，可以选择打印它们
        if unmatched_paths > 0:
            unmatched = set(filter_df['path']) - set(complete_df['path'])
            print("\n前5个未匹配的路径示例:")
            for path in list(unmatched)[:5]:
                print(path)
            
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")

# 使用示例
if __name__ == "__main__":
    # 替换为实际的文件路径
    complete_csv = "dataloader/droid/480p24fps/droid_subset_caption.csv"  # 包含所有字段的CSV
    filter_csv = "dataloader/droid/480p24fps/droid_right.csv"      # 只包含path的CSV
    output_csv = "dataloader/droid/480p24fps/droid_right_caption.csv"  # 输出文件路径
    
    filter_csv_by_path(complete_csv, filter_csv, output_csv)