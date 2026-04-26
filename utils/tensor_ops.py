#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    : tensor_ops.py
@Time    : 2026/04/26
@Author  : Bai Xueqiong
@Email   : amybx2006@sina.com
@Description: 张量操作工具函数
"""

# Copyright (c) 2026 by Bai Xueqiong, All Rights Reserved.
# Licensed under the MIT License

from typing import Union, Tuple

import torch
import torch.nn.functional as F


def cus_sample(
        feat: torch.Tensor,
        size: Union[Tuple[int, int], None] = None,
        scale_factor: Union[float, None] = None,
        mode: str = 'bilinear',
        align_corners: bool = True
) -> torch.Tensor:
    """
    自定义上采样函数

    Args:
        feat: 输入特征图 (B, C, H, W)
        size: 目标尺寸 (H, W)，与 scale_factor 二选一
        scale_factor: 缩放因子，与 size 二选一
        mode: 插值模式，默认 'bilinear'
        align_corners: 是否对齐角点，默认 True

    Returns:
        上采样后的特征图

    Raises:
        ValueError: 如果 size 和 scale_factor 都未提供或都提供

    Examples:
        >>> x = torch.randn(2, 64, 32, 32)
        >>> # 使用 scale_factor
        >>> y = cus_sample(x, scale_factor=2)  # (2, 64, 64, 64)
        >>> # 使用 size
        >>> z = cus_sample(x, size=(64, 64))   # (2, 64, 64, 64)
    """
    if size is None and scale_factor is None:
        raise ValueError("Must provide either 'size' or 'scale_factor'")

    if size is not None and scale_factor is not None:
        raise ValueError("Cannot provide both 'size' and 'scale_factor'")

    kwargs = {}
    if size is not None:
        kwargs['size'] = size
    if scale_factor is not None:
        kwargs['scale_factor'] = scale_factor

    return F.interpolate(
        feat,
        mode=mode,
        align_corners=align_corners,
        **kwargs
    )


def resize_as(feat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    将特征图调整为与目标张量相同的尺寸

    Args:
        feat: 输入特征图 (B, C, H, W)
        target: 目标张量 (B, C', H', W')

    Returns:
        调整后的特征图 (B, C, H', W')

    Example:
        >>> x = torch.randn(2, 64, 32, 32)
        >>> target = torch.randn(2, 128, 64, 64)
        >>> y = resize_as(x, target)  # (2, 64, 64, 64)
    """
    return cus_sample(feat, size=target.shape[2:])
