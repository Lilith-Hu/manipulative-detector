import pandas as pd
import os
from sklearn.model_selection import train_test_split

# 📂 数据路径 (使用基于文件位置的路径)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 获取项目根目录
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset_bert.csv")
TRAIN_PATH = os.path.join(BASE_DIR, "data", "train.csv")
VAL_PATH = os.path.join(BASE_DIR, "data", "val.csv")


def load_and_check_data(file_path):
    """加载 CSV 文件、检查字段，并进行数据清洗"""
    df = pd.read_csv(file_path, encoding="utf-8-sig", sep=";")  # 👈 指定分隔符为分号

    # ✅ 检查必要字段
    required_columns = ["text", "category"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"❌ 缺少必要字段: {col}")

    # 🧹 数据清洗
    df = df.dropna()  # 移除空值
    df["text"] = df["text"].str.strip()  # 去除文本中的多余空格

    print("📋 数据预览：")
    print(df.head())
    print("\n📊 数据统计信息：")
    print(df.info())

    return df


def split_and_save_data(df, train_path, val_path):
    """将数据集划分为训练集和验证集，并保存为 CSV 文件"""
    # 🔀 数据集划分 (80% 训练集, 20% 验证集)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # 💾 保存数据集
    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    val_df.to_csv(val_path, index=False, encoding="utf-8-sig")

    print("\n✅ 训练集和验证集已保存至 data 文件夹")


if __name__ == "__main__":
    # 📂 文件路径检查
    print("📂 当前工作目录:", os.getcwd())
    print("📂 目标路径:", DATA_PATH)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"❌ 文件未找到: {DATA_PATH}")

    df = load_and_check_data(DATA_PATH)

    # 数据集划分与保存
    split_and_save_data(df, TRAIN_PATH, VAL_PATH)
    print("\n✅ 数据预处理完成！")
