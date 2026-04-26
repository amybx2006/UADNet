#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    : convnext.py
@Time    : 2024/04/12 16:18:10
@Author  : Bai Xueqiong
@Email   : amybx2006@sina.com
@Description: ConvNeXt 模型实现
    
    ConvNeXt: A ConvNet for the 2020s
    Paper: https://arxiv.org/pdf/2201.03545.pdf

"""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the MIT License
# ========================================
# Original Paper: A ConvNet for the 2020s
# ========================================

from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model


# ==================== 基础模块 ====================

class Block(nn.Module):
    """
    ConvNeXt 基础块
    Args:
        dim: 输入通道数
        drop_path: 随机深度率，默认 0.0
        layer_scale_init_value: Layer Scale 初始值，默认 1e-6
    """
    
    def __init__(
        self,
        dim: int,
        drop_path: float = 0.,
        layer_scale_init_value: float = 1e-6
    ):
        super().__init__()
        
        # 深度可分离卷积 (7x7)
        self.dwconv = nn.Conv2d(
            dim, dim,
            kernel_size=7,
            padding=3,
            groups=dim  # 深度可分离
        )
        
        # LayerNorm (channels_last 格式)
        self.norm = LayerNorm(dim, eps=1e-6)
        
        # 第一个 1x1 卷积（扩展通道）
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        
        # GELU 激活函数
        self.act = nn.GELU()
        
        # 第二个 1x1 卷积（恢复通道）
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
        # Layer Scale（可学习的缩放因子）
        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(
                layer_scale_init_value * torch.ones(dim),
                requires_grad=True
            )
        else:
            self.gamma = None
        
        # Drop Path（随机深度）
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 (N, C, H, W)
            
        Returns:
            输出特征 (N, C, H, W)
        """
        identity = x
        
        # 深度可分离卷积
        x = self.dwconv(x)
        
        # (N, C, H, W) -> (N, H, W, C)
        x = x.permute(0, 2, 3, 1)
        
        # LayerNorm
        x = self.norm(x)
        
        # 1x1 卷积 + GELU + 1x1 卷积
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        # Layer Scale
        if self.gamma is not None:
            x = self.gamma * x
        
        # (N, H, W, C) -> (N, C, H, W)
        x = x.permute(0, 3, 1, 2)
        
        # 残差连接 + Drop Path
        x = identity + self.drop_path(x)
        
        return x


class LayerNorm(nn.Module):
    """
    LayerNorm 支持两种数据格式
    
    - channels_last: (batch_size, height, width, channels)
    - channels_first: (batch_size, channels, height, width)
    
    Args:
        normalized_shape: 归一化的形状（通道数）
        eps: 数值稳定性参数，默认 1e-6
        data_format: 数据格式，默认 "channels_last"
    """
    
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        data_format: str = "channels_last"
    ):
        super().__init__()
        
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(
                f"Invalid data_format: {data_format}. "
                f"Expected 'channels_last' or 'channels_first'."
            )
        
        self.normalized_shape = (normalized_shape,)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征
            
        Returns:
            归一化后的特征
        """
        if self.data_format == "channels_last":
            # 使用 PyTorch 内置的 LayerNorm
            return F.layer_norm(
                x, self.normalized_shape,
                self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            # 手动实现 LayerNorm（channels_first 格式）
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


# ==================== ConvNeXt 主网络 ====================

class ConvNeXt(nn.Module):
    """
    ConvNeXt 网络

    Args:
        in_chans: 输入图像通道数，默认 3
        num_classes: 分类类别数，默认 1000
        depths: 每个 Stage 的 Block 数量，默认 [3, 3, 9, 3]
        dims: 每个 Stage 的特征维度，默认 [96, 192, 384, 768]
        drop_path_rate: 随机深度率，默认 0.0
        layer_scale_init_value: Layer Scale 初始值，默认 1e-6
        head_init_scale: 分类头初始化缩放因子，默认 1.0
    """
    
    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 1000,
        depths: List[int] = None,
        dims: List[int] = None,
        drop_path_rate: float = 0.,
        layer_scale_init_value: float = 1e-6,
        head_init_scale: float = 1.
    ):
        super().__init__()
        
        # 默认配置（ConvNeXt-Tiny）
        if depths is None:
            depths = [3, 3, 9, 3]
        if dims is None:
            dims = [96, 192, 384, 768]
        
        # ==================== 下采样层 ====================
        self.downsample_layers = nn.ModuleList()
        
        # Stem: 4x4 卷积，stride=4
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        
        # 3 个中间下采样层：2x2 卷积，stride=2
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2)
            )
            self.downsample_layers.append(downsample_layer)
        
        # ==================== 特征提取阶段 ====================
        self.stages = nn.ModuleList()
        
        # 计算每个 Block 的 drop_path_rate（线性递增）
        dp_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]
        
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    Block(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]
        
        # ==================== 分类头 ====================
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)
        
        # ==================== 初始化权重 ====================
        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)
    
    def _init_weights(self, m: nn.Module):
        """
        初始化权重
        
        Args:
            m: 模块
        """
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        提取多尺度特征（用于下游任务）
        
        Args:
            x: 输入图像 (B, 3, H, W)
            
        Returns:
            多尺度特征列表：
                - outs[0]: (B, dims[0], H/4, W/4)
                - outs[1]: (B, dims[1], H/8, W/8)
                - outs[2]: (B, dims[2], H/16, W/16)
                - outs[3]: (B, dims[3], H/32, W/32)
        """
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            outs.append(x)
        return outs
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播（分类任务）
        
        Args:
            x: 输入图像 (B, 3, H, W)
            
        Returns:
            分类 logits (B, num_classes)
        """
        # 提取特征
        features = self.forward_features(x)
        x = features[-1]
        
        # 全局平均池化
        x = self.norm(x.mean([-2, -1]))  # (B, C, H, W) -> (B, C)
        
        # 分类头
        x = self.head(x)
        
        return x


# ==================== 预训练模型 URL ====================

MODEL_URLS = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}


# ==================== 模型构建函数 ====================

@register_model
def convnext_tiny(pretrained: bool = False, in_22k: bool = False, **kwargs) -> ConvNeXt:
    """
    ConvNeXt-Tiny 模型
    
    配置：depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]
    参数量：~28M
    
    Args:
        pretrained: 是否加载预训练权重
        in_22k: 是否使用 ImageNet-22K 预训练权重
        **kwargs: 其他参数
        
    Returns:
        ConvNeXt-Tiny 模型
    """
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    
    if pretrained:
        url = MODEL_URLS['convnext_tiny_22k'] if in_22k else MODEL_URLS['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url, map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    
    return model


@register_model
def convnext_small(pretrained: bool = False, in_22k: bool = False, **kwargs) -> ConvNeXt:
    """
    ConvNeXt-Small 模型
    
    配置：depths=[3, 3, 27, 3], dims=[96, 192, 384, 768]
    参数量：~50M
    
    Args:
        pretrained: 是否加载预训练权重
        in_22k: 是否使用 ImageNet-22K 预训练权重
        **kwargs: 其他参数
        
    Returns:
        ConvNeXt-Small 模型
    """
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    
    if pretrained:
        url = MODEL_URLS['convnext_small_22k'] if in_22k else MODEL_URLS['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    
    return model


@register_model
def convnext_base(pretrained: bool = False, in_22k: bool = False, **kwargs) -> ConvNeXt:
    """
    ConvNeXt-Base 模型
    
    配置：depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024]
    参数量：~89M
    
    Args:
        pretrained: 是否加载预训练权重
        in_22k: 是否使用 ImageNet-22K 预训练权重
        **kwargs: 其他参数
        
    Returns:
        ConvNeXt-Base 模型
    """
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    
    if pretrained:
        url = MODEL_URLS['convnext_base_22k'] if in_22k else MODEL_URLS['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    
    return model


@register_model
def convnext_large(pretrained: bool = False, in_22k: bool = False, **kwargs) -> ConvNeXt:
    """
    ConvNeXt-Large 模型
    
    配置：depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536]
    参数量：~198M
    
    Args:
        pretrained: 是否加载预训练权重
        in_22k: 是否使用 ImageNet-22K 预训练权重
        **kwargs: 其他参数
        
    Returns:
        ConvNeXt-Large 模型
    """
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    
    if pretrained:
        url = MODEL_URLS['convnext_large_22k'] if in_22k else MODEL_URLS['convnext_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    
    return model


@register_model
def convnext_xlarge(pretrained: bool = False, in_22k: bool = False, **kwargs) -> ConvNeXt:
    """
    ConvNeXt-XLarge 模型
    
    配置：depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048]
    参数量：~350M
    
    注意：只有 ImageNet-22K 预训练权重可用
    
    Args:
        pretrained: 是否加载预训练权重
        in_22k: 是否使用 ImageNet-22K 预训练权重
        **kwargs: 其他参数
        
    Returns:
        ConvNeXt-XLarge 模型
        
    Raises:
        AssertionError: 如果 pretrained=True 但 in_22k=False
    """
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    
    if pretrained:
        if not in_22k:
            raise ValueError(
                "Only ImageNet-22K pre-trained ConvNeXt-XLarge is available. "
                "Please set in_22k=True."
            )
        url = MODEL_URLS['convnext_xlarge_22k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    
    return model


# ==================== 测试函数 ====================

def test_convnext():
    """测试 ConvNeXt 模型"""
    print('='*70)
    print('Testing ConvNeXt Models')
    print('='*70)
    
    # 测试配置
    batch_size = 2
    input_size = 224
    num_classes = 1000
    
    # 测试所有模型变体
    models = {
        'ConvNeXt-Tiny': convnext_tiny(pretrained=False, num_classes=num_classes),
        'ConvNeXt-Small': convnext_small(pretrained=False, num_classes=num_classes),
        'ConvNeXt-Base': convnext_base(pretrained=False, num_classes=num_classes),
        'ConvNeXt-Large': convnext_large(pretrained=False, num_classes=num_classes),
        'ConvNeXt-XLarge': convnext_xlarge(pretrained=False, num_classes=num_classes),
    }
    
    x = torch.randn(batch_size, 3, input_size, input_size)
    
    for name, model in models.items():
        model.eval()
        
        print(f'\n{name}:')
        
        # 测试分类
        with torch.no_grad():
            output = model(x)
        print(f'  Classification output: {tuple(output.shape)}')
        
        # 测试特征提取
        with torch.no_grad():
            features = model.forward_features(x)
        print(f'  Feature extraction:')
        for i, feat in enumerate(features):
            print(f'    Level {i}: {tuple(feat.shape)}')
        
        # 统计参数量
        num_params = sum(p.numel() for p in model.parameters())
        print(f'  Parameters: {num_params:,} ({num_params/1e6:.1f}M)')
    
    print('\n' + '='*70)
    print('All tests passed! ✓')
    print('='*70)


if __name__ == '__main__':
    test_convnext()
