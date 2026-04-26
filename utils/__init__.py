#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    : __init__.py
@Description: Utils Package - 工具函数和数据加载器
"""

# Copyright (c) 2026 by Bai Xueqiong, All Rights Reserved.
# Licensed under the MIT License

# 通用工具函数
from .utils import (
    clip_gradient,
    poly_lr,
    AvgMeter,
    calc_params,
    CalParams
)

# 张量操作
from .tensor_ops import (
    cus_sample,
    resize_as
)

# 边缘数据加载器
from .dataloader_edge import (
    CamObjDataset,
    test_dataset,
    test_loader_faster,
    test_dataset_new,
    get_loader
)

# 通用数据加载器
from .dataloader import (
    PolypDataset,
    TestDataset,
    test_dataset as test_dataset_polyp
)

# 优化器
from .AdaX import AdaX, AdaXW

__version__ = '1.0.0'

__all__ = [
    # 工具函数
    'clip_gradient',
    'poly_lr',
    'AvgMeter',
    'calc_params',
    'CalParams',

    # 张量操作
    'cus_sample',
    'resize_as',

    # 边缘数据加载器
    'CamObjDataset',
    'test_dataset',
    'test_loader_faster',
    'test_dataset_new',
    'get_loader',

    # 通用数据加载器
    'PolypDataset',
    'TestDataset',
    'test_dataset_polyp',

    # 优化器
    'AdaX',
    'AdaXW',
]
