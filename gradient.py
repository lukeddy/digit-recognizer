"""
gradient.py — 数值梯度计算
----------------------------
使用中心差分法计算函数对各参数的数值梯度。
该方法无需了解反向传播，但计算速度较慢（O(n) 次函数调用）。
后续学习反向传播后可替换为解析梯度。
"""

import numpy as np


def _numerical_gradient_1d(f, x: np.ndarray) -> np.ndarray:
    """
    对一维向量 x 计算函数 f 的数值梯度（中心差分）。

    中心差分公式：f'(x) ≈ [f(x+h) - f(x-h)] / (2h)
    精度优于前向差分，误差为 O(h²)。
    """
    h     = 1e-4
    grad  = np.zeros_like(x)

    for i in range(x.size):
        tmp  = x[i]

        x[i] = tmp + h
        fxh1 = f(x)

        x[i] = tmp - h
        fxh2 = f(x)

        grad[i] = (fxh1 - fxh2) / (2 * h)
        x[i]    = tmp       # 还原原始值，避免影响后续计算

    return grad


def numerical_gradient(f, X: np.ndarray) -> np.ndarray:
    """
    计算函数 f 对参数矩阵 X 的数值梯度。

    支持 1D（向量）和 2D（矩阵）输入：
    - 1D：直接计算
    - 2D：对每一行分别计算（适用于权重矩阵）

    参数
    ----
    f : 目标函数，接受与 X 相同形状的数组，返回 scalar
    X : 参数数组（权重矩阵或偏置向量）

    返回
    ----
    grad : 与 X 形状相同的梯度数组
    """
    if X.ndim == 1:
        return _numerical_gradient_1d(f, X)
    else:
        grad = np.zeros_like(X)
        for i, x in enumerate(X):
            grad[i] = _numerical_gradient_1d(f, x)
        return grad
