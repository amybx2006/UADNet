#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    : Encoder.py
@Time    : 2026/04/16 10:21:43
@Author  : Bai Xueqiong
@Email   : amybx2006@sina.com
@Description: 编码器模块
"""

# Copyright (c) 2026 Bai Xueqiong, All Rights Reserved.
# Licensed under the MIT License
# ========================================
# Project: UAD-Net Encoder
# ========================================

from typing import Tuple

import torch
import torch.nn as nn

from model.convnext import convnext_base


# ==================== ConvNeXt 编码器 ====================

class Encoder_convnext(nn.Module):
    """
    基于 ConvNeXt-Base 的编码器

    使用预训练的 ConvNeXt-Base 模型提取多尺度特征。
    ConvNeXt 是一个现代化的卷积网络，具有以下特点：
    - 深度可分离卷积
    - 大卷积核 (7x7)
    - 层归一化
    - GELU 激活函数

    特征尺度：
    - x0: (B, 128, H/4, W/4)   - 浅层特征，细节丰富
    - x1: (B, 256, H/8, W/8)   - 中层特征
    - x2: (B, 512, H/16, W/16) - 深层特征
    - x3: (B, 1024, H/32, W/32) - 最深层特征，语义信息丰富

    Args:
        pretrained: 是否使用预训练权重，默认 True
        num_classes: 分类头的类别数（仅用于加载预训练权重），默认 1000

    Example:
        >>> encoder = Encoder_convnext()
        >>> x = torch.randn(2, 3, 352, 352)
        >>> x0, x1, x2, x3 = encoder(x)
        >>> print(x0.shape)  # torch.Size([2, 128, 88, 88])
    """

    def __init__(self, pretrained: bool = True, num_classes: int = 1000):
        super().__init__()

        # 加载 ConvNeXt-Base 预训练模型
        self.convnext = convnext_base(pretrained=pretrained, num_classes=num_classes)

        # 冻结 BatchNorm 层（可选，用于微调）
        # self._freeze_bn()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入图像 (B, 3, H, W)

        Returns:
            Tuple of:
                - x0: 浅层特征 (B, 128, H/4, W/4)
                - x1: 中层特征 (B, 256, H/8, W/8)
                - x2: 深层特征 (B, 512, H/16, W/16)
                - x3: 最深层特征 (B, 1024, H/32, W/32)
        """
        # 提取多尺度特征
        features = self.convnext.forward_features(x)

        x0 = features[0]  # (B, 128, H/4, W/4)
        x1 = features[1]  # (B, 256, H/8, W/8)
        x2 = features[2]  # (B, 512, H/16, W/16)
        x3 = features[3]  # (B, 1024, H/32, W/32)

        return x0, x1, x2, x3

    def _freeze_bn(self):
        """冻结所有 BatchNorm 层（可选，用于微调时保持统计信息）"""
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False


# ==================== Res2Net 编码器（可选） ====================

class Encoder_Res2net(nn.Module):
    """
    基于 Res2Net-50 的编码器（可选）

    使用预训练的 Res2Net-50 模型提取多尺度特征。
    Res2Net 通过在残差块内部构建分层连接来增强多尺度表示能力。

    注意：需要安装 Res2Net 库并取消注释相关导入

    特征尺度：
    - x0: (B, 256, H/4, W/4)
    - x1: (B, 512, H/8, W/8)
    - x2: (B, 1024, H/16, W/16)
    - x3: (B, 2048, H/32, W/32)

    Args:
        pretrained: 是否使用预训练权重，默认 True

    Example:
        >>> encoder = Encoder_Res2net()
        >>> x = torch.randn(2, 3, 352, 352)
        >>> x0, x1, x2, x3 = encoder(x)
        >>> print(x0.shape)  # torch.Size([2, 256, 88, 88])
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # 取消注释以下行以使用 Res2Net
        # from lib.Res2Net_v1b import res2net50_v1b_26w_4s
        # self.resnet = res2net50_v1b_26w_4s(pretrained=pretrained)

        raise NotImplementedError(
            "Res2Net encoder is not implemented. "
            "Please install Res2Net library and uncomment the import statement."
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入图像 (B, 3, H, W)

        Returns:
            Tuple of:
                - x0: 浅层特征 (B, 256, H/4, W/4)
                - x1: 中层特征 (B, 512, H/8, W/8)
                - x2: 深层特征 (B, 1024, H/16, W/16)
                - x3: 最深层特征 (B, 2048, H/32, W/32)
        """
        # 取消注释以下代码以使用 Res2Net
        # # Stem
        # x = self.resnet.conv1(x)
        # x = self.resnet.bn1(x)
        # x = self.resnet.relu(x)
        # x = self.resnet.maxpool(x)  # (B, 64, H/4, W/4)

        # # ResNet stages
        # x0 = self.resnet.layer1(x)   # (B, 256, H/4, W/4)
        # x1 = self.resnet.layer2(x0)  # (B, 512, H/8, W/8)
        # x2 = self.resnet.layer3(x1)  # (B, 1024, H/16, W/16)
        # x3 = self.resnet.layer4(x2)  # (B, 2048, H/32, W/32)

        # return x0, x1, x2, x3

        raise NotImplementedError("Res2Net encoder is not implemented.")


# ==================== 工厂函数 ====================

def build_encoder(encoder_type: str = 'convnext', pretrained: bool = True) -> nn.Module:
    """
    构建编码器的工厂函数

    Args:
        encoder_type: 编码器类型，可选 'convnext' 或 'res2net'
        pretrained: 是否使用预训练权重

    Returns:
        编码器模型

    Raises:
        ValueError: 如果编码器类型不支持

    Example:
        >>> encoder = build_encoder('convnext', pretrained=True)
    """
    encoder_type = encoder_type.lower()

    if encoder_type == 'convnext':
        return Encoder_convnext(pretrained=pretrained)
    elif encoder_type == 'res2net':
        return Encoder_Res2net(pretrained=pretrained)
    else:
        raise ValueError(
            f"Unsupported encoder type: {encoder_type}. "
            f"Supported types: ['convnext', 'res2net']"
        )


# ==================== 测试函数 ====================

def test_encoder(
    encoder: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 352, 352),
    expected_shapes: Tuple[Tuple[int, ...], ...] = None
):
    """
    测试编码器的前向传播和输出形状

    Args:
        encoder: 编码器模型
        input_size: 输入尺寸 (B, C, H, W)
        expected_shapes: 期望的输出形状列表
    """
    print('='*70)
    print(f'Testing {encoder.__class__.__name__}')
    print('='*70)

    # 设置为评估模式
    encoder.eval()

    # 创建测试输入
    x = torch.randn(*input_size)
    print(f'\n📥 Input shape: {tuple(x.shape)}')

    # 前向传播
    with torch.no_grad():
        x0, x1, x2, x3 = encoder(x)

    # 打印输出形状
    print(f'\n📤 Output shapes:')
    print(f'   x0 (1/4):  {tuple(x0.shape)}')
    print(f'   x1 (1/8):  {tuple(x1.shape)}')
    print(f'   x2 (1/16): {tuple(x2.shape)}')
    print(f'   x3 (1/32): {tuple(x3.shape)}')

    # 验证输出形状
    if expected_shapes is not None:
        shapes = [x0.shape, x1.shape, x2.shape, x3.shape]
        for i, (actual, expected) in enumerate(zip(shapes, expected_shapes)):
            assert actual == torch.Size(expected), \
                f"Shape mismatch at x{i}: expected {expected}, got {tuple(actual)}"
        print('\n✓ All shape assertions passed!')

    print('='*70 + '\n')


def main():
    """主测试函数"""
    # 测试 ConvNeXt 编码器
    print('Testing ConvNeXt Encoder...\n')

    encoder_convnext = Encoder_convnext(pretrained=False)  # 测试时不加载预训练权重

    test_encoder(
        encoder_convnext,
        input_size=(1, 3, 352, 352),
        expected_shapes=[
            (1, 128, 88, 88),   # x0: 352/4 = 88
            (1, 256, 44, 44),   # x1: 352/8 = 44
            (1, 512, 22, 22),   # x2: 352/16 = 22
            (1, 1024, 11, 11)   # x3: 352/32 = 11
        ]
    )

    # 测试不同输入尺寸
    print('Testing with different input size (384x384)...\n')

    test_encoder(
        encoder_convnext,
        input_size=(2, 3, 384, 384),
        expected_shapes=[
            (2, 128, 96, 96),   # x0: 384/4 = 96
            (2, 256, 48, 48),   # x1: 384/8 = 48
            (2, 512, 24, 24),   # x2: 384/16 = 24
            (2, 1024, 12, 12)   # x3: 384/32 = 12
        ]
    )

    # 测试工厂函数
    print('Testing factory function...\n')
    encoder = build_encoder('convnext', pretrained=False)
    print(f'✓ Successfully built encoder: {encoder.__class__.__name__}\n')

    print('='*70)
    print('All tests passed successfully! ✓')
    print('='*70)


if __name__ == '__main__':
    main()
