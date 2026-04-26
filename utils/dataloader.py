#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    : dataloader.py
@Time    : 2026/04/26
@Author  : Bai Xueqiong
@Email   : amybx2006@sina.com
@Description: 通用数据加载器
"""

# Copyright (c) 2026 by Bai Xueqiong, All Rights Reserved.
# Licensed under the MIT License

import os
from pathlib import Path
from typing import Tuple

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

# 支持的图像格式
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')
GT_EXTENSIONS = ('.png', '.tif')


class PolypDataset(data.Dataset):
    """
    息肉分割数据集

    Args:
        image_root: 图像根目录
        gt_root: Ground Truth 根目录
        trainsize: 训练图像尺寸

    Example:
        >>> dataset = PolypDataset(
        ...     'data/polyp/images',
        ...     'data/polyp/masks',
        ...     trainsize=352
        ... )
        >>> image, gt = dataset[0]
    """

    def __init__(self, image_root: str, gt_root: str, trainsize: int):
        self.trainsize = trainsize

        # 使用 Path 对象
        image_root = Path(image_root)
        gt_root = Path(gt_root)

        # 获取文件列表
        self.images = sorted([
            str(image_root / f) for f in os.listdir(image_root)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        ])

        self.gts = sorted([
            str(gt_root / f) for f in os.listdir(gt_root)
            if f.lower().endswith('.png')
        ])

        # 过滤不匹配的文件
        self.filter_files()
        self.size = len(self.images)

        # 图像变换
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取单个样本

        Args:
            index: 样本索引

        Returns:
            Tuple of (image, gt)
        """
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])

        image = self.img_transform(image)
        gt = self.gt_transform(gt)

        return image, gt

    def filter_files(self) -> None:
        """过滤尺寸不匹配的文件"""
        if len(self.images) != len(self.gts):
            print(f'⚠ Warning: File count mismatch - '
                  f'images: {len(self.images)}, gts: {len(self.gts)}')

        images, gts = [], []

        for img_path, gt_path in zip(self.images, self.gts):
            try:
                img = Image.open(img_path)
                gt = Image.open(gt_path)

                if img.size == gt.size:
                    images.append(img_path)
                    gts.append(gt_path)
            except Exception as e:
                print(f'⚠ Warning: Failed to load {img_path}: {str(e)}')
                continue

        self.images = images
        self.gts = gts

    def rgb_loader(self, path: str) -> Image.Image:
        """加载 RGB 图像"""
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path: str) -> Image.Image:
        """加载二值图像"""
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self) -> int:
        return self.size


class TestDataset:
    """
    测试数据集

    Args:
        image_root: 图像根目录
        gt_root: Ground Truth 根目录
        testsize: 测试图像尺寸
    """

    def __init__(self, image_root: str, gt_root: str, testsize: int):
        self.testsize = testsize

        # 使用 Path 对象
        image_root = Path(image_root)
        gt_root = Path(gt_root)

        # 获取文件列表
        self.images = sorted([
            str(image_root / f) for f in os.listdir(image_root)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        ])

        self.gts = sorted([
            str(gt_root / f) for f in os.listdir(gt_root)
            if f.lower().endswith(GT_EXTENSIONS)
        ])

        # 图像变换
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.size = len(self.images)
        self.index = 0

    def load_data(self) -> Tuple[torch.Tensor, Image.Image, str]:
        """
        加载单张图像

        Returns:
            Tuple of (image, gt, name)
        """
        image = self.rgb_loader(self.images[self.index])
        gt = self.binary_loader(self.gts[self.index])

        # 获取文件名
        name = Path(self.images[self.index]).name
        if name.lower().endswith('.jpg'):
            name = name.replace('.jpg', '.png')

        # 应用变换
        image = self.transform(image).unsqueeze(0)

        self.index += 1
        return image, gt, name

    def rgb_loader(self, path: str) -> Image.Image:
        """加载 RGB 图像"""
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path: str) -> Image.Image:
        """加载二值图像"""
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


# 保持向后兼容
test_dataset = TestDataset


def get_loader(
        image_root: str,
        gt_root: str,
        batchsize: int,
        trainsize: int,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True
) -> data.DataLoader:
    """
    获取训练数据加载器

    Args:
        image_root: 图像根目录
        gt_root: Ground Truth 根目录
        batchsize: 批次大小
        trainsize: 训练图像尺寸
        shuffle: 是否打乱数据，默认 True
        num_workers: 工作进程数，默认 4
        pin_memory: 是否固定内存，默认 True

    Returns:
        PyTorch DataLoader
    """
    dataset = PolypDataset(image_root, gt_root, trainsize)

    return data.DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
