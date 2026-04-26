#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    : utils.py
@Time    : 2026/04/26
@Author  : Bai Xueqiong
@Email   : amybx2006@sina.com
@Description: 通用工具函数集合
"""

# Copyright (c) 2026 by Bai Xueqiong, All Rights Reserved.
# Licensed under the MIT License

from typing import Optional

import torch
import numpy as np
from thop import profile, clever_format


def clip_gradient(optimizer: torch.optim.Optimizer, grad_clip: float) -> None:
    """
    梯度裁剪，防止梯度爆炸

    Args:
        optimizer: PyTorch 优化器
        grad_clip: 裁剪阈值

    Example:
        >>> optimizer = torch.optim.Adam(model.parameters())
        >>> clip_gradient(optimizer, 0.5)
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def poly_lr(
        optimizer: torch.optim.Optimizer,
        init_lr: float,
        curr_iter: int,
        max_iter: int,
        power: float = 0.9
) -> None:
    """
    多项式学习率衰减策略

    公式: lr = init_lr * (1 - curr_iter / max_iter) ^ power

    Args:
        optimizer: PyTorch 优化器
        init_lr: 初始学习率
        curr_iter: 当前迭代次数
        max_iter: 最大迭代次数
        power: 衰减指数，默认 0.9

    Example:
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        >>> poly_lr(optimizer, 1e-4, 100, 1000)
    """
    lr = init_lr * (1 - float(curr_iter) / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AvgMeter:
    """
    平均值记录器，用于记录训练过程中的损失等指标

    支持滑动窗口平均，避免早期数据对后期统计的影响

    Args:
        num: 滑动窗口大小，默认 40

    Example:
        >>> meter = AvgMeter()
        >>> for i in range(100):
        ...     meter.update(i * 0.1)
        >>> print(meter.show())  # 显示最近 40 个值的平均
    """

    def __init__(self, num: int = 40):
        self.num = num
        self.reset()

    def reset(self) -> None:
        """重置所有统计信息"""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
        self.losses = []

    def update(self, val: float, n: int = 1) -> None:
        """
        更新统计信息

        Args:
            val: 当前值
            n: 样本数量，默认 1
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self) -> float:
        """
        返回滑动窗口内的平均值

        Returns:
            最近 num 个值的平均值
        """
        if len(self.losses) == 0:
            return 0.0

        start_idx = max(len(self.losses) - self.num, 0)
        recent_losses = self.losses[start_idx:]

        if isinstance(recent_losses[0], torch.Tensor):
            return torch.mean(torch.stack(recent_losses)).item()
        else:
            return np.mean(recent_losses)


def calc_params(
        model: torch.nn.Module,
        input_tensor: torch.Tensor,
        verbose: bool = True
) -> tuple:
    """
    计算模型参数量和 FLOPs

    需要安装 thop 库：pip install thop

    Args:
        model: PyTorch 模型
        input_tensor: 输入张量示例
        verbose: 是否打印结果，默认 True

    Returns:
        Tuple of (flops, params)

    Example:
        >>> model = MyModel()
        >>> x = torch.randn(1, 3, 352, 352)
        >>> flops, params = calc_params(model, x)
        [Statistics Information]
          FLOPs:  12.345G
          Params: 23.456M
    """
    flops, params = profile(model, inputs=(input_tensor,))
    flops_str, params_str = clever_format([flops, params], "%.3f")

    if verbose:
        print('[Statistics Information]')
        print(f'  FLOPs:  {flops_str}')
        print(f'  Params: {params_str}')

    return flops, params


# 保持向后兼容
CalParams = calc_params
