"""
dataset.py — 数据加载与预处理
------------------------------
负责从原始 CSV 文件读取 MNIST 数据，进行数据集划分与特征归一化，
并以无状态函数形式对外暴露，方便单元测试和复用。
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from config import DATA_PATH, DATA_CONFIG


def load_mnist(
    data_path: str = DATA_PATH,
    test_size: float = DATA_CONFIG['test_size'],
    random_state: int = DATA_CONFIG['random_state'],
    normalize: bool = DATA_CONFIG['normalize'],
) -> tuple:
    """
    加载 MNIST 手写数字数据集。

    参数
    ----
    data_path    : CSV 文件路径
    test_size    : 测试集占比
    random_state : 随机种子
    normalize    : 是否对特征做 MinMax 归一化（归一到 [0, 1]）

    返回
    ----
    (x_train, x_test, y_train, y_test) — 均为 numpy ndarray:
        x_train / x_test : shape (n, 784), dtype float64
        y_train / y_test : shape (n,),     dtype int64（类别标签 0-9）
    """
    # 1. 读取原始 CSV
    data = pd.read_csv(data_path)

    # 2. 分离特征与标签
    X = data.drop(columns='label').values.astype(np.float64)
    y = data['label'].values

    # 3. 划分训练集 / 测试集
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 4. 特征归一化（在训练集上 fit，避免数据泄露）
    if normalize:
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_test  = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test


def describe_dataset(x_train, x_test, y_train, y_test) -> None:
    """打印数据集基本信息，方便调试和 EDA。"""
    print("=" * 45)
    print("  数据集信息")
    print("=" * 45)
    print(f"  训练集特征: {x_train.shape}  标签: {y_train.shape}")
    print(f"  测试集特征: {x_test.shape}   标签: {y_test.shape}")
    print(f"  特征值范围: [{x_train.min():.4f}, {x_train.max():.4f}]")
    print(f"  类别分布:   {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print("=" * 45)
