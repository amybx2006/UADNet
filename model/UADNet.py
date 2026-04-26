#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    : UADNet.py
@Time    : 2026/04/18 15:22:40
@Author  : Bai Xueqiong
@Email   : amybx2006@sina.com
@Description: UAD-Net
"""

# Copyright (c) 2026 Bai Xueqiong, All Rights Reserved.
# Licensed under the MIT License
# ========================================
# Project: UAD-Net
# ========================================

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.Encoder import Encoder_convnext as Encoder


# ==================== 辅助函数 ====================

def custom_upsample(feat: torch.Tensor, **kwargs) -> torch.Tensor:

    if not kwargs:
        raise ValueError("Must provide either 'size' or 'scale_factor'")

    valid_keys = {'size', 'scale_factor'}
    if not set(kwargs.keys()).issubset(valid_keys):
        raise ValueError(f"Invalid keys. Expected {valid_keys}, got {set(kwargs.keys())}")

    return F.interpolate(feat, **kwargs, mode='bilinear', align_corners=True)


# ==================== 基础卷积模块 ====================

class BasicConv2d(nn.Module):
    """
    基础卷积模块：Conv2d + BatchNorm + ReLU

    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小
        stride: 步长，默认 1
        padding: 填充，默认 0
        dilation: 膨胀率，默认 1
        relu: 是否使用 ReLU 激活，默认 False
        bn: 是否使用 BatchNorm，默认 True
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        relu: bool = False,
        bn: bool = True
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=not bn  # 如果有 BN，则不需要 bias
        )
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入特征 (B, C_in, H, W)

        Returns:
            输出特征 (B, C_out, H', W')
        """
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Conv1x1(nn.Module):
    """
    1x1 卷积模块：Conv2d(1x1) + BatchNorm + ReLU

    用于通道数调整和特征变换

    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入特征 (B, C_in, H, W)

        Returns:
            输出特征 (B, C_out, H, W)
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ConvBNR(nn.Module):
    """
    标准卷积模块：Conv2d + BatchNorm + ReLU

    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小，默认 3
        stride: 步长，默认 1
        dilation: 膨胀率，默认 1
        bias: 是否使用偏置，默认 False
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = False
    ):
        super().__init__()

        padding = dilation * (kernel_size - 1) // 2

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.block(x)


# ==================== 注意力模块 ====================

class MSCA(nn.Module):
    """
    多尺度通道注意力模块 (Multi-Scale Channel Attention)

    结合局部和全局上下文信息生成通道注意力权重：
    - 局部分支：使用 1x1 卷积提取局部通道关系
    - 全局分支：使用全局平均池化提取全局通道统计

    Args:
        channels: 输入通道数，默认 64
        reduction: 通道缩减比例，默认 4
    """

    def __init__(self, channels: int = 64, reduction: int = 4):
        super().__init__()

        reduced_channels = channels // reduction

        # 局部注意力分支
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=1),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels)
        )

        # 全局注意力分支
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced_channels, kernel_size=1),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入特征 (B, C, H, W)

        Returns:
            注意力权重 (B, C, H, W)，范围 [0, 1]
        """
        local_att = self.local_att(x)
        global_att = self.global_att(x)

        # 融合局部和全局注意力
        combined_att = local_att + global_att
        attention_weight = self.sigmoid(combined_att)

        return attention_weight


# ==================== 特征融合模块 ====================

class ESFM(nn.Module):
    """
    边缘感知特征融合模块 (Edge-aware Selective Fusion Module)

    使用多尺度通道注意力自适应融合两个不同层级的特征：
    1. 上采样对齐特征尺寸
    2. 特征相加生成融合特征
    3. MSCA 生成注意力权重
    4. 加权融合两个特征
    5. 拼接原始特征并降维

    Args:
        in_channels: 输入通道数（拼接后），默认 128
        out_channels: 输出通道数，默认 64
        scale_factor: 上采样倍数，None 表示不上采样
    """

    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 64,
        scale_factor: Optional[float] = None
    ):
        super().__init__()

        self.msca = MSCA(channels=out_channels)
        self.scale_factor = scale_factor
        self.conv = BasicConv2d(
            in_channels, out_channels,
            kernel_size=3, padding=1, relu=True
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 当前层特征 (B, C, H, W)
            y: 上一层特征 (B, C, H', W')

        Returns:
            融合后的特征 (B, C_out, H, W)
        """
        # 上采样对齐尺寸
        if self.scale_factor is not None and self.scale_factor != 1:
            y = custom_upsample(y, scale_factor=self.scale_factor)

        # 特征相加
        fused = x + y

        # 生成注意力权重
        attention = self.msca(fused)

        # 加权融合：x 和 y 的加权组合
        weighted_fusion = x * attention + y * (1 - attention)

        # 拼接原始特征 x
        concatenated = torch.cat([weighted_fusion, x], dim=1)

        # 卷积降维
        output = self.conv(concatenated)
        output = F.relu(output)

        return output


# ==================== 特征增强模块 ====================

class TFEM(nn.Module):
    """
    三分支特征增强模块 (Triple-branch Feature Enhancement Module)

    使用不同感受野的卷积分支提取多尺度特征：
    - 分支 0: 1x1 卷积（最小感受野）
    - 分支 1: 1x1 -> 1x3+3x1 -> 3x3(dilation=3)（中等感受野）
    - 分支 2: 1x1 -> 1x5+5x1 -> 3x3(dilation=5)（较大感受野）
    - 分支 3: 1x1 -> 1x7+7x1 -> 3x3(dilation=7)（最大感受野）

    最后通过残差连接增强特征表达

    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)

        # 分支 0: 1x1 卷积（最小感受野）
        self.branch_small = BasicConv2d(in_channels, out_channels, kernel_size=1)

        # 分支 1: 1x1 -> 1x3+3x1 -> 3x3(dilation=3)
        self.branch_medium = nn.Sequential(
            BasicConv2d(in_channels, out_channels, kernel_size=1),
            BasicConv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channels, out_channels, kernel_size=3, padding=3, dilation=3)
        )

        # 分支 2: 1x1 -> 1x5+5x1 -> 3x3(dilation=5)
        self.branch_large = nn.Sequential(
            BasicConv2d(in_channels, out_channels, kernel_size=1),
            BasicConv2d(out_channels, out_channels, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channels, out_channels, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channels, out_channels, kernel_size=3, padding=5, dilation=5)
        )

        # 分支 3: 1x1 -> 1x7+7x1 -> 3x3(dilation=7)
        self.branch_xlarge = nn.Sequential(
            BasicConv2d(in_channels, out_channels, kernel_size=1),
            BasicConv2d(out_channels, out_channels, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channels, out_channels, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channels, out_channels, kernel_size=3, padding=7, dilation=7)
        )

        # 融合卷积
        self.conv_cat = BasicConv2d(4 * out_channels, out_channels, kernel_size=3, padding=1)
        self.conv_res = BasicConv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入特征 (B, C_in, H, W)

        Returns:
            增强后的特征 (B, C_out, H, W)
        """
        # 多分支特征提取
        feat_small = self.branch_small(x)
        feat_medium = self.branch_medium(x)
        feat_large = self.branch_large(x)
        feat_xlarge = self.branch_xlarge(x)

        # 拼接多尺度特征
        feat_cat = torch.cat([feat_small, feat_medium, feat_large, feat_xlarge], dim=1)
        feat_fused = self.conv_cat(feat_cat)

        # 残差连接
        residual = self.conv_res(x)
        output = self.relu(feat_fused + residual)

        return output


class ResidualConv(nn.Module):
    """
    残差卷积模块

    两层 3x3 卷积 + 残差连接，用于增强深层特征

    Args:
        channels: 通道数，默认 1024
    """

    def __init__(self, channels: int = 1024):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入特征 (B, C, H, W)

        Returns:
            输出特征 (B, C, H, W)
        """
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        # 残差连接
        out = out + identity
        out = self.relu(out)

        return out


# ==================== 边缘预测模块 ====================

class CGEPB(nn.Module):
    """
    跨层边缘预测分支 (Cross-layer Global Edge Prediction Branch)

    融合浅层和深层特征预测边缘图：
    1. 降维浅层和深层特征
    2. 上采样深层特征对齐尺寸
    3. 拼接融合
    4. 卷积预测边缘

    Args:
        shallow_channels: 浅层特征通道数，默认 128
        deep_channels: 深层特征通道数，默认 1024
        mid_channels: 中间通道数，默认 256
        reduced_channels: 降维后通道数，默认 64
    """

    def __init__(
        self,
        shallow_channels: int = 128,
        deep_channels: int = 1024,
        mid_channels: int = 256,
        reduced_channels: int = 64
    ):
        super().__init__()

        self.reduce_shallow = Conv1x1(shallow_channels, reduced_channels)
        self.reduce_deep = Conv1x1(deep_channels, mid_channels)

        self.fusion_block = nn.Sequential(
            ConvBNR(mid_channels + reduced_channels, mid_channels, kernel_size=3),
            ConvBNR(mid_channels, mid_channels, kernel_size=3),
            nn.Conv2d(mid_channels, 1, kernel_size=1)
        )

    def forward(
        self,
        deep_feat: torch.Tensor,
        shallow_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            deep_feat: 深层特征 (B, C_deep, H_d, W_d)
            shallow_feat: 浅层特征 (B, C_shallow, H_s, W_s)

        Returns:
            边缘预测图 (B, 1, H_s, W_s)
        """
        target_size = shallow_feat.shape[2:]

        # 降维
        shallow_reduced = self.reduce_shallow(shallow_feat)
        deep_reduced = self.reduce_deep(deep_feat)

        # 上采样对齐
        deep_upsampled = F.interpolate(
            deep_reduced,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )

        # 拼接融合
        fused = torch.cat([deep_upsampled, shallow_reduced], dim=1)
        edge_map = self.fusion_block(fused)

        return edge_map


# ==================== 主网络 ====================

class UADNet(nn.Module):
    """
    UAD-Net: Uncertainty-Aware Deep Network for Camouflaged Object Detection

    基于边缘感知和多尺度特征融合的伪装目标检测网络

    网络结构：
        1. Encoder: ConvNeXt 编码器提取多尺度特征 [C1, C2, C3, C4]
        2. ResidualConv: 残差卷积增强最深层特征 C4
        3. CGEPB: 跨层边缘预测分支（融合 C4 和 C1）
        4. TFEM: 三分支特征增强模块（4 个不同尺度）
        5. ESFM: 边缘感知特征融合模块（自底向上融合）
        6. 深度监督: 多尺度输出 [O0, O1, O2] + 边缘输出

    特点：
        - 多尺度特征提取和融合
        - 边缘感知的特征选择
        - 深度监督训练
        - 残差连接增强

    Args:
        encoder_channels: 编码器各层通道数，默认 (128, 256, 512, 1024)
        decoder_channels: 解码器通道数，默认 64

    Input:
        x: 输入图像 (B, 3, H, W)

    Output:
        Tuple of:
            - output_0: 最终输出 (B, 1, H, W)
            - output_1: 中间输出 1 (B, 1, H, W)
            - output_2: 中间输出 2 (B, 1, H, W)
            - edge: 边缘预测 (B, 1, H, W)
    """

    def __init__(
        self,
        encoder_channels: Tuple[int, int, int, int] = (128, 256, 512, 1024),
        decoder_channels: int = 64
    ):
        super().__init__()

        c1, c2, c3, c4 = encoder_channels

        # ==================== 编码器 ====================
        self.encoder = Encoder()

        # ==================== 残差增强 ====================
        self.res_conv = ResidualConv(channels=c4)

        # ==================== 边缘预测分支 ====================
        self.edge_branch = CGEPB(
            shallow_channels=c1,
            deep_channels=c4,
            mid_channels=256,
            reduced_channels=decoder_channels
        )

        # ==================== 三分支特征增强模块 ====================
        self.tfem_0 = TFEM(c1, decoder_channels)
        self.tfem_1 = TFEM(c2, decoder_channels)
        self.tfem_2 = TFEM(c3, decoder_channels)
        self.tfem_3 = TFEM(c4, decoder_channels)

        # ==================== 边缘感知特征融合模块 ====================
        self.esfm_3 = ESFM(
            in_channels=decoder_channels * 2,
            out_channels=decoder_channels,
            scale_factor=None  # 不上采样
        )
        self.esfm_2 = ESFM(
            in_channels=decoder_channels * 2,
            out_channels=decoder_channels,
            scale_factor=None
        )
        self.esfm_1 = ESFM(
            in_channels=decoder_channels * 2,
            out_channels=decoder_channels,
            scale_factor=None
        )
        self.esfm_0 = ESFM(
            in_channels=decoder_channels * 2,
            out_channels=decoder_channels,
            scale_factor=None
        )

        # ==================== 通道降维 ====================
        self.channel_reducer = Conv1x1(c4, decoder_channels)

        # ==================== 深度监督输出 ====================
        self.output_3 = Conv1x1(decoder_channels, 1)
        self.output_2 = Conv1x1(decoder_channels, 1)
        self.output_1 = Conv1x1(decoder_channels, 1)
        self.output_final = Conv1x1(decoder_channels, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入图像 (B, 3, H, W)

        Returns:
            Tuple of:
                - output_0: 最终输出 (B, 1, H, W)
                - output_1: 中间输出 1 (B, 1, H, W)
                - output_2: 中间输出 2 (B, 1, H, W)
                - edge: 边缘预测 (B, 1, H, W)
        """
        # ==================== 编码器提取多尺度特征 ====================
        feat_0, feat_1, feat_2, feat_3 = self.encoder(x)
        # feat_0: (B, 128, H/4, W/4)
        # feat_1: (B, 256, H/8, W/8)
        # feat_2: (B, 512, H/16, W/16)
        # feat_3: (B, 1024, H/32, W/32)

        # ==================== 残差增强最深层特征 ====================
        feat_3_enhanced = self.res_conv(feat_3)

        # ==================== 边缘预测 ====================
        edge_map = self.edge_branch(feat_3_enhanced, feat_0)
        edge_att = torch.sigmoid(edge_map)

        # ==================== 三分支特征增强 ====================
        feat_0_enhanced = self.tfem_0(feat_0)  # (B, 64, H/4, W/4)
        feat_1_enhanced = self.tfem_1(feat_1)  # (B, 64, H/8, W/8)
        feat_2_enhanced = self.tfem_2(feat_2)  # (B, 64, H/16, W/16)
        feat_3_enhanced = self.tfem_3(feat_3)  # (B, 64, H/32, W/32)

        # ==================== 降维 ====================
        feat_reduced = self.channel_reducer(feat_3_enhanced)  # (B, 64, H/32, W/32)

        # ==================== 自底向上特征融合 ====================
        # Level 3: (B, 64, H/32, W/32)
        fused_3 = self.esfm_3(feat_3_enhanced, feat_reduced)
        fused_3_up = custom_upsample(fused_3, scale_factor=2)  # -> (B, 64, H/16, W/16)

        # Level 2: (B, 64, H/16, W/16)
        fused_2 = self.esfm_2(feat_2_enhanced, fused_3_up)
        fused_2_up = custom_upsample(fused_2, scale_factor=2)  # -> (B, 64, H/8, W/8)

        # Level 1: (B, 64, H/8, W/8)
        fused_1 = self.esfm_1(feat_1_enhanced, fused_2_up)
        fused_1_up = custom_upsample(fused_1, scale_factor=2)  # -> (B, 64, H/4, W/4)

        # Level 0: (B, 64, H/4, W/4)
        fused_0 = self.esfm_0(feat_0_enhanced, fused_1_up)

        # ==================== 深度监督输出（上采样到原图尺寸） ====================
        output_3 = custom_upsample(self.output_3(fused_3), scale_factor=16)  # H/32 -> H/2
        output_2 = custom_upsample(self.output_2(fused_2), scale_factor=8)   # H/16 -> H/2
        output_1 = custom_upsample(self.output_1(fused_1), scale_factor=4)   # H/8 -> H/2
        output_0 = custom_upsample(self.output_final(fused_0), scale_factor=2)  # H/4 -> H/2

        # ==================== 边缘输出（上采样到原图尺寸） ====================
        edge_output = F.interpolate(
            edge_att,
            scale_factor=4,
            mode='bilinear',
            align_corners=True
        )

        return output_0, output_1, output_2, edge_output


# 保持向后兼容
Net = UADNet


# ==================== 辅助函数 ====================

def count_parameters(model: nn.Module) -> int:
    """
    统计模型可训练参数量

    Args:
        model: PyTorch 模型

    Returns:
        可训练参数总数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(model: nn.Module, input_size: Tuple[int, int, int, int] = (2, 3, 384, 384)):
    """
    打印模型信息

    Args:
        model: PyTorch 模型
        input_size: 输入尺寸 (B, C, H, W)
    """
    print('='*70)
    print('UAD-Net Model Information')
    print('='*70)

    # 统计参数
    num_params = count_parameters(model)
    print(f'\n Model Parameters:')
    print(f'   Total: {num_params:,} ({num_params/1e6:.2f}M)')

    # 测试前向传播
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    print(f'\n Device: {device}')
    print(f'\n Input size: {input_size}')

    # 创建测试输入
    x = torch.randn(*input_size).to(device)

    # 前向传播
    with torch.no_grad():
        output_0, output_1, output_2, edge = model(x)

    # 输出结果
    print(f'\n Output shapes:')
    print(f'   Final output (O0):     {tuple(output_0.shape)}')
    print(f'   Intermediate 1 (O1):   {tuple(output_1.shape)}')
    print(f'   Intermediate 2 (O2):   {tuple(output_2.shape)}')
    print(f'   Edge prediction:       {tuple(edge.shape)}')

    print('\n' + '='*70)
    print('✓ Model testing completed successfully!')
    print('='*70 + '\n')


# ==================== 主函数 ====================

def main():
    """测试模型"""
    # 创建模型
    model = UADNet()

    # 打印模型信息
    print_model_info(model, input_size=(2, 3, 384, 384))

    # 可选：保存模型结构
    # torch.save(model.state_dict(), 'uadnet_init.pth')


if __name__ == '__main__':
    main()
