"""
main.py — 程序入口
--------------------
将所有模块串联起来，完成完整的训练流程：
  1. 加载数据
  2. 初始化模型
  3. 初始化优化器
  4. 创建训练器并开始训练
  5. 可视化结果

只需运行此文件即可启动训练：
    cd test/digit_recognizer
    python main.py
"""

import sys
import os

# 确保可以正确导入本目录下的模块
sys.path.insert(0, os.path.dirname(__file__))

from config import TRAIN_CONFIG, LOG_CONFIG
from dataset import load_mnist, describe_dataset
from model import TwoLayerNet
from optimizer import build_optimizer
from trainer import Trainer
from visualizer import plot_training_results


def main():
    # ────────────────────────────────────────────────────────── #
    #  1. 加载数据
    # ────────────────────────────────────────────────────────── #
    print("正在加载 MNIST 数据...")
    x_train, x_test, y_train, y_test = load_mnist()
    describe_dataset(x_train, x_test, y_train, y_test)

    # ────────────────────────────────────────────────────────── #
    #  2. 初始化模型
    # ────────────────────────────────────────────────────────── #
    model = TwoLayerNet()

    # ────────────────────────────────────────────────────────── #
    #  3. 初始化优化器
    #     当前：SGD（随机梯度下降）
    #     后续：在 config.TRAIN_CONFIG['optimizer'] 中切换为
    #            'momentum' 或 'adam' 即可，代码无需改动
    # ────────────────────────────────────────────────────────── #
    optimizer = build_optimizer(
        optimizer_name=TRAIN_CONFIG['optimizer'],
        learning_rate=TRAIN_CONFIG['learning_rate'],
    )

    # ────────────────────────────────────────────────────────── #
    #  4. 创建训练器并开始训练
    # ────────────────────────────────────────────────────────── #
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )
    history = trainer.train()

    # ────────────────────────────────────────────────────────── #
    #  5. 可视化训练结果
    # ────────────────────────────────────────────────────────── #
    plot_training_results(history)


if __name__ == '__main__':
    main()
