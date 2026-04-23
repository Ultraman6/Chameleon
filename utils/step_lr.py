from typing import List
import math

class ConstLR:
    def __call__(self, lr):
        return lr

class StepLRforWRN:
    def __init__(self, learning_rate: float, total_epochs: int):
        """_summary_

        Args:
            learning_rate (float): _description_
            total_epochs (int): _description_
        """
        self.total_epochs = total_epochs
        self.base_lr = learning_rate

    def __call__(self, optimizer, epoch):
        if epoch < self.total_epochs * 3/10:
            lr = self.base_lr
        elif epoch < self.total_epochs * 6/10:
            lr = self.base_lr * 0.2
        # elif epoch < self.total_epochs * 8/10:
        #     lr = self.base * 0.2 ** 2
        else:
            lr = self.base_lr * 0.2 ** 2

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return lr

class MultiStepLR:
    def __init__(self, learning_rate: float, milestones: List[int], gamma: float):
        """_summary_

        Args:
            learning_rate (float): _description_
            milestones (List[int]): _description_
            gamma (float): _description_
        """
        self.milestones = milestones
        self.base_lr = learning_rate
        self.gamma = gamma
        
    def __call__(self, optimizer, epoch):
        lr = self.base_lr
        for milestone in self.milestones:
            if epoch >= milestone - 1:
                lr *= self.gamma

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return lr

class CosineAnnealingLR:
    def __init__(self, learning_rate: float, T_max: int, eta_min: float = 0.0):
        """
        余弦退火学习率调度器

        Args:
            learning_rate (float): 基础学习率
            total_epochs (int): 总训练轮次
            eta_min (float): 最小学习率，默认为0
        """
        self.base_lr = learning_rate
        self.T_max = T_max
        self.eta_min = eta_min

    def __call__(self, optimizer, epoch):
        # 余弦退火公式
        lr = self.eta_min + (self.base_lr - self.eta_min) * (
                1 + math.cos(math.pi * epoch / self.T_max)
        ) / 2

        # 更新优化器学习率
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return lr


class PolynomialLR:
    def __init__(self, learning_rate: float, T_max: int, power: float = 1.0):
        """
        多项式学习率调度器

        Args:
            learning_rate (float): 基础学习率
            total_epochs (int): 总训练轮次
            power (float): 多项式指数，默认为1.0（线性衰减）
        """
        self.base_lr = learning_rate
        self.T_max = T_max
        self.power = power

    def __call__(self, optimizer, epoch):
        # 多项式衰减公式
        lr = self.base_lr * (1 - epoch / self.T_max) ** self.power

        # 更新优化器学习率
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return lr