import pandas as pd
import os

# 📂 设置 BERT 数据路径
DATA_PATH = "../data/dataset_bert.csv"  # 确保路径正确

def load_and_check_data(file_path):
    """加载 CSV 文件并检查格式"""
    # 读取 CSV 文件
    df = pd.read_csv(file_path, encoding="utf-8-sig")

    # 显示数据前 5 行
    print("📋 数据预览：")
    print(df.head())

    # 检查字段是否正确
    required_columns = ["text", "category"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"❌ 缺少必要字段: {col}")

    # 打印每个字段的非空数据量
    print("\n📦 字段数据量统计：")
    print(df.count())

    return df


if __name__ == "__main__":
    # 确保文件存在
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"❌ 文件未找到: {DATA_PATH}")

    # 加载数据
    df = load_and_check_data(DATA_PATH)
    print("\n✅ 数据加载完成！")
