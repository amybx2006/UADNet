#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    : AdaX.py
@Time    : 2026/04/26
@Author  : Bai Xueqiong
@Email   : amybx2006@sina.com
@Description: AdaX 优化器实现

    AdaX: Adaptive Gradient Descent with Exponential Long Term Memory

    参考文献：
    - Adam: A Method for Stochastic Optimization
      https://arxiv.org/abs/1412.6980
    - On the Convergence of Adam and Beyond
      https://openreview.net/forum?id=ryQu7f-RZ
"""

# Copyright (c) 2026 by Bai Xueqiong, All Rights Reserved.
# Licensed under the MIT License

import math
from typing import Optional, Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer


class AdaX(Optimizer):
    """
    AdaX 优化器

    特点：
    - 使用指数长期记忆更新二阶矩
    - L2 正则化
    - 偏差修正

    Args:
        params: 可迭代的参数或参数组字典
        lr: 学习率，默认 1.5e-3
        betas: 用于计算梯度及其平方的运行平均值的系数，默认 (0.9, 1e-4)
        eps: 数值稳定性参数，默认 1e-12
        weight_decay: L2 正则化系数，默认 5e-4

    Example:
        >>> optimizer = AdaX(model.parameters(), lr=1.5e-3)
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()
    """

    def __init__(
            self,
            params: Iterable,
            lr: float = 1.5e-3,
            betas: Tuple[float, float] = (0.9, 1e-4),
            eps: float = 1e-12,
            weight_decay: float = 5e-4
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """
        执行单步优化

        Args:
            closure: 重新评估模型并返回损失的闭包函数

        Returns:
            损失值（如果提供了 closure）
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        'AdaX does not support sparse gradients, '
                        'please consider SparseAdam instead'
                    )

                state = self.state[p]

                # 状态初始化
                if len(state) == 0:
                    state['step'] = 0
                    # 一阶矩估计（梯度的指数移动平均）
                    state['exp_avg'] = torch.zeros_like(p)
                    # 二阶矩估计（梯度平方的指数移动平均）
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1

                # 权重衰减（L2 正则化）
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # 更新一阶矩估计
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # 更新二阶矩估计（AdaX 特有）
                exp_avg_sq.mul_(1 + beta2).addcmul_(grad, grad, value=beta2)

                # 计算分母
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                # 偏差修正
                bias_correction2 = (1 + beta2) ** state['step'] - 1
                step_size = group['lr'] * math.sqrt(bias_correction2)

                # 更新参数
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class AdaXW(Optimizer):
    """
    AdaXW 优化器（带解耦权重衰减）

    与 AdaX 的区别：
    - 使用解耦权重衰减（类似 AdamW）
    - 权重衰减直接应用于参数，而非梯度

    Args:
        params: 可迭代的参数或参数组字典
        lr: 学习率，默认 5e-3
        betas: 用于计算梯度及其平方的运行平均值的系数，默认 (0.9, 1e-4)
        eps: 数值稳定性参数，默认 1e-12
        weight_decay: 权重衰减系数，默认 5e-2

    Example:
        >>> optimizer = AdaXW(model.parameters(), lr=5e-3, weight_decay=5e-2)
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()
    """

    def __init__(
            self,
            params: Iterable,
            lr: float = 5e-3,
            betas: Tuple[float, float] = (0.9, 1e-4),
            eps: float = 1e-12,
            weight_decay: float = 5e-2
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """
        执行单步优化

        Args:
            closure: 重新评估模型并返回损失的闭包函数

        Returns:
            损失值（如果提供了 closure）
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        'AdaX does not support sparse gradients, '
                        'please consider SparseAdam instead'
                    )

                state = self.state[p]

                # 状态初始化
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1

                # 更新一阶矩估计
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # 更新二阶矩估计
                exp_avg_sq.mul_(1 + beta2).addcmul_(grad, grad, value=beta2)

                # 计算分母
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                # 偏差修正
                bias_correction2 = (1 + beta2) ** state['step'] - 1
                step_size = group['lr'] * math.sqrt(bias_correction2)

                # 解耦权重衰减 + 参数更新
                p.mul_(1 - group['lr'] * group['weight_decay'])
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
