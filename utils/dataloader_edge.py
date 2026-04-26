#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    : dataloader_edge.py
@Time    : 2026/04/26
@Author  : Bai Xueqiong
@Email   : amybx2006@sina.com
@Description: 训练和测试数据加载器（带边缘标注）
"""

# Copyright (c) 2026 by Bai Xueqiong, All Rights Reserved.
# Licensed under the MIT License

import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

# 支持的图像格式
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')
GT_EXTENSIONS = ('.png', '.tif')


class CamObjDataset(data.Dataset):
    """
    伪装目标检测训练数据集（带边缘标注）

    数据增强：
    - 随机水平翻转
    - 边缘膨胀（5x5 核）

    Args:
        image_root: 图像根目录
        gt_root: Ground Truth 根目录
        edge_root: 边缘标注根目录
        trainsize: 训练图像尺寸

    Example:
        >>> dataset = CamObjDataset(
        ...     'data/train/images',
        ...     'data/train/masks',
        ...     'data/train/edges',
        ...     trainsize=352
        ... )
        >>> image, gt, edge = dataset[0]
    """

    def __init__(
            self,
            image_root: str,
            gt_root: str,
            edge_root: str,
            trainsize: int
    ):
        self.trainsize = trainsize

        # 使用 Path 对象处理路径
        image_root = Path(image_root)
        gt_root = Path(gt_root)
        edge_root = Path(edge_root)

        # 获取文件列表
        self.images = sorted([
            str(image_root / f) for f in os.listdir(image_root)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        ])

        self.gts = sorted([
            str(gt_root / f) for f in os.listdir(gt_root)
            if f.lower().endswith('.png')
        ])

        self.edges = sorted([
            str(edge_root / f) for f in os.listdir(edge_root)
            if f.lower().endswith('.png')
        ])

        # 过滤不匹配的文件
        self.filter_files()
        self.size = len(self.images)

        # 边缘膨胀核
        self.kernel = np.ones((5, 5), np.uint8)

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

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取单个样本

        Args:
            index: 样本索引

        Returns:
            Tuple of:
                - image: 预处理后的图像 (3, H, W)
                - gt: Ground Truth (1, H, W)
                - edge: 边缘标注 (1, H, W)
        """
        # 随机水平翻转
        flip = transforms.RandomHorizontalFlip(p=np.random.rand())

        # 加载图像
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        edge = cv2.imread(self.edges[index], cv2.IMREAD_GRAYSCALE)

        # 应用翻转
        image = flip(image)
        gt = flip(gt)

        # 边缘膨胀
        edge = cv2.dilate(edge, self.kernel, iterations=1)
        edge = Image.fromarray(edge)
        edge = flip(edge)

        # 应用变换
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        edge = self.gt_transform(edge)

        return image, gt, edge

    def filter_files(self) -> None:
        """过滤尺寸不匹配的文件"""
        if len(self.images) != len(self.gts) or len(self.images) != len(self.edges):
            print(f'⚠ Warning: File count mismatch - '
                  f'images: {len(self.images)}, '
                  f'gts: {len(self.gts)}, '
                  f'edges: {len(self.edges)}')

        images, gts, edges = [], [], []

        for img_path, gt_path, edge_path in zip(self.images, self.gts, self.edges):
            try:
                img = Image.open(img_path)
                gt = Image.open(gt_path)
                edge = Image.open(edge_path)

                if img.size == gt.size == edge.size:
                    images.append(img_path)
                    gts.append(gt_path)
                    edges.append(edge_path)
            except Exception as e:
                print(f'⚠ Warning: Failed to load {img_path}: {str(e)}')
                continue

        self.images = images
        self.gts = gts
        self.edges = edges

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


class test_dataset:
    """
    测试数据集（基础版本）

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
            if f.lower().endswith('.png')
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

        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self) -> Tuple[torch.Tensor, Image.Image, str]:
        """加载单张图像"""
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)

        gt = self.binary_loader(self.gts[self.index])

        name = Path(self.images[self.index]).name
        if name.lower().endswith('.jpg'):
            name = name.replace('.jpg', '.png')

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


class test_loader_faster(data.Dataset):
    """
    快速测试数据加载器（支持批处理）

    Args:
        image_root: 图像根目录
        testsize: 测试图像尺寸
    """

    def __init__(self, image_root: str, testsize: int):
        self.testsize = testsize

        image_root = Path(image_root)

        self.images = sorted([
            str(image_root / f) for f in os.listdir(image_root)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        ])

        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.size = len(self.images)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        """获取单个样本"""
        image = self.rgb_loader(self.images[index])
        image = self.transform(image)
        img_name = self.images[index]

        return image, img_name

    def rgb_loader(self, path: str) -> Image.Image:
        """加载 RGB 图像"""
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __len__(self) -> int:
        return self.size


class test_dataset_new:
    """
    测试数据集（增强版本，兼容 DGNet）

    提供原始尺寸的图像用于后处理

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

    def load_data(self) -> Tuple[torch.Tensor, Image.Image, str, np.ndarray]:
        """
        加载单张图像

        Returns:
            Tuple of:
                - image: 预处理后的图像 (1, 3, H, W)
                - gt: Ground Truth (原始尺寸)
                - name: 文件名
                - image_for_post: 用于后处理的图像数组
        """
        # 加载图像
        image = self.rgb_loader(self.images[self.index])
        gt = self.binary_loader(self.gts[self.index])

        # 获取文件名
        name = Path(self.images[self.index]).name
        if name.lower().endswith('.jpg'):
            name = name.replace('.jpg', '.png')
        elif name.lower().endswith('.jpeg'):
            name = name.replace('.jpeg', '.png')

        # 后处理用图像（调整到 GT 尺寸）
        image_for_post = self.rgb_loader(self.images[self.index])
        image_for_post = image_for_post.resize(gt.size)

        # 应用变换
        image = self.transform(image).unsqueeze(0)

        # 更新索引（循环）
        self.index = (self.index + 1) % self.size

        return image, gt, name, np.array(image_for_post)

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


def get_loader(
        image_root: str,
        gt_root: str,
        edge_root: str,
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
        edge_root: 边缘标注根目录
        batchsize: 批次大小
        trainsize: 训练图像尺寸
        shuffle: 是否打乱数据，默认 True
        num_workers: 工作进程数，默认 4
        pin_memory: 是否固定内存，默认 True

    Returns:
        PyTorch DataLoader

    Example:
        >>> loader = get_loader(
        ...     'data/train/images',
        ...     'data/train/masks',
        ...     'data/train/edges',
        ...     batchsize=16,
        ...     trainsize=352
        ... )
        >>> for images, gts, edges in loader:
        ...     # 训练代码
        ...     pass
    """
    dataset = CamObjDataset(image_root, gt_root, edge_root, trainsize)

    return data.DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
