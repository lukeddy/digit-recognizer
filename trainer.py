"""
trainer.py — 训练器
---------------------
封装"训练循环"的全部逻辑，包括：
  - Mini-batch 随机采样
  - 梯度计算与参数更新
  - 训练 loss / 准确率记录
  - 日志打印

设计哲学
--------
Trainer 只依赖"接口"（model.gradient / model.loss / model.accuracy），
不关心网络内部实现细节 —— 换模型时 Trainer 无需修改。
"""

import numpy as np
from model import TwoLayerNet
from optimizer import BaseOptimizer
from config import TRAIN_CONFIG, LOG_CONFIG


class TrainHistory:
    """
    保存训练过程中记录的指标，与绘图模块解耦。

    属性
    ----
    train_loss_list : 每次迭代的训练 loss
    train_acc_list  : 每个 Epoch 结束时的训练集准确率
    test_acc_list   : 每个 Epoch 结束时的测试集准确率
    """

    def __init__(self):
        self.train_loss_list: list[float] = []
        self.train_acc_list:  list[float] = []
        self.test_acc_list:   list[float] = []


class Trainer:
    """
    模型训练器。

    用法示例
    --------
    >>> trainer = Trainer(model, optimizer, x_train, y_train, x_test, y_test)
    >>> history = trainer.train()
    """

    def __init__(
        self,
        model: TwoLayerNet,
        optimizer: BaseOptimizer,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test:  np.ndarray,
        y_test:  np.ndarray,
        batch_size:    int  = TRAIN_CONFIG['batch_size'],
        num_epochs:    int  = TRAIN_CONFIG['num_epochs'],
        grad_method:   str  = TRAIN_CONFIG['grad_method'],
        print_loss_every: int = LOG_CONFIG['print_loss_every'],
    ):
        self.model      = model
        self.optimizer  = optimizer
        self.x_train    = x_train
        self.y_train    = y_train
        self.x_test     = x_test
        self.y_test     = y_test

        self.batch_size       = batch_size
        self.num_epochs       = num_epochs
        self.grad_method      = grad_method
        self.print_loss_every = print_loss_every

        # 计算训练迭代总次数
        # 如果不理解，查看 gemini 的解释： https://gemini.google.com/share/10e61eab5a3b
        n = x_train.shape[0]
        self.iters_per_epoch = int(np.ceil(n / batch_size))
        self.total_iters     = self.iters_per_epoch * num_epochs

    # ---------------------------------------------------------------- #
    #  主训练方法
    # ---------------------------------------------------------------- #
    def train(self) -> TrainHistory:
        """
        执行完整训练流程，返回 TrainHistory 对象。
        """
        history = TrainHistory()
        n = self.x_train.shape[0]

        self._print_train_header()

        for iteration in range(self.total_iters):
            # ① Mini-batch 随机采样
            batch_idx  = np.random.choice(n, self.batch_size)
            x_batch    = self.x_train[batch_idx] # 从训练集中取特征
            t_batch    = self.y_train[batch_idx] # 从训练集中取对应的标签

            # ② 计算梯度
            grads = self.model.gradient(x_batch, t_batch, method=self.grad_method)

            # ③ 更新参数（由优化器决定策略）
            self.optimizer.step(self.model.params, grads)

            # ④ 记录当前 batch 损失
            loss = self.model.loss(x_batch, t_batch)
            history.train_loss_list.append(loss)

            # ⑤ 按频率打印 loss
            if iteration % self.print_loss_every == 0:
                epoch_num = iteration // self.iters_per_epoch + 1
                print(f"  [iter {iteration:>5d} | epoch {epoch_num}/{self.num_epochs}]  loss: {loss:.6f}")

            # ⑥ 每个 Epoch 结束时评估准确率
            if iteration % self.iters_per_epoch == 0:
                self._evaluate_epoch(iteration, history)

        # 训练结束后进行最终评估
        self._evaluate_final(history)
        return history

    # ---------------------------------------------------------------- #
    #  辅助方法
    # ---------------------------------------------------------------- #
    def _evaluate_epoch(self, iteration: int, history: TrainHistory) -> None:
        """计算并记录当前 Epoch 的训练/测试准确率。"""
        train_acc = self.model.accuracy(self.x_train, self.y_train)
        test_acc  = self.model.accuracy(self.x_test,  self.y_test)
        history.train_acc_list.append(train_acc)
        history.test_acc_list.append(test_acc)
        epoch_num = iteration // self.iters_per_epoch + 1
        print(f"\n{'─'*50}")
        print(f"  Epoch {epoch_num:>2d}/{self.num_epochs}  "
              f"train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")
        print(f"{'─'*50}\n")

    def _evaluate_final(self, history: TrainHistory) -> None:
        """训练结束，打印最终准确率。"""
        train_acc = self.model.accuracy(self.x_train, self.y_train)
        test_acc  = self.model.accuracy(self.x_test,  self.y_test)
        print("\n" + "=" * 50)
        print("  训练完成！最终结果：")
        print(f"  最终训练集准确率: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"  最终测试集准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")
        print("=" * 50)

    def _print_train_header(self) -> None:
        """打印训练开始前的配置摘要。"""
        print("\n" + "=" * 50)
        print("  开始训练 —— 两层神经网络 (MNIST 手写数字识别)")
        print("=" * 50)
        print(f"  样本数量  : {self.x_train.shape[0]} (训练) / {self.x_test.shape[0]} (测试)")
        print(f"  Epochs    : {self.num_epochs}")
        print(f"  Batch Size: {self.batch_size}")
        print(f"  总迭代次数: {self.total_iters}")
        print(f"  梯度方式  : {self.grad_method}")
        print("=" * 50 + "\n")
