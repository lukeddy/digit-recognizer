"""
visualizer.py — 训练结果可视化
--------------------------------
从 TrainHistory 读取指标，绘制：
  1. 训练 Loss 曲线（每次迭代）
  2. 训练集 / 测试集 准确率曲线（每个 Epoch）

可选将图表保存到磁盘（由 config.LOG_CONFIG['save_plot'] 控制）。
"""

import os
import matplotlib.pyplot as plt
import matplotlib
from trainer import TrainHistory
from config import LOG_CONFIG

# 设置中文字体，避免乱码
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def plot_training_results(history: TrainHistory, save: bool = LOG_CONFIG['save_plot']) -> None:
    """
    绘制训练 Loss 曲线 + 准确率曲线（双子图）。

    参数
    ----
    history : TrainHistory 对象
    save    : True 则将图表保存为 PNG 文件
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('MNIST 手写数字识别 — 训练过程', fontsize=15, fontweight='bold')

    # ── 子图1：Loss 曲线 ──────────────────────────────────────── #
    ax1.plot(history.train_loss_list, color='#E74C3C', linewidth=1.2, alpha=0.85, label='训练 Loss')
    ax1.set_title('训练 Loss（每次迭代）')
    ax1.set_xlabel('迭代次数 (Iterations)')
    ax1.set_ylabel('交叉熵损失 (Cross‑Entropy Loss)')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.4)

    # ── 子图2：准确率曲线 ─────────────────────────────────────── #
    epochs = range(1, len(history.train_acc_list) + 1)
    ax2.plot(epochs, history.train_acc_list, 'o-', color='#2980B9',
             linewidth=1.8, markersize=5, label='训练集准确率')
    ax2.plot(epochs, history.test_acc_list, 's--', color='#27AE60',
             linewidth=1.8, markersize=5, label='测试集准确率')
    ax2.set_title('准确率（每个 Epoch）')
    ax2.set_xlabel('训练轮次 (Epochs)')
    ax2.set_ylabel('准确率 (Accuracy)')
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()

    if save:
        output_dir = LOG_CONFIG['plot_output_dir']
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, 'training_results.png')
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n  📊 训练曲线已保存至: {save_path}")

    plt.show()
