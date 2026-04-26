#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    : MyTest.py
@Time    : 2026/04/26 13:05:00
@Author  : Bai Xueqiong
@Email   : amybx2006@sina.com
@Description: UAD-Net 测试脚本
"""

# Copyright (c) 2026 Bai Xueqiong, All Rights Reserved.
# Licensed under the MIT License
# ========================================
# Project: UAD-Net Testing
# ========================================

import os
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

import torch
import torch.nn.functional as F
import numpy as np
import imageio
from skimage import img_as_ubyte
from tqdm import tqdm

from model.UADNet import Net
from utils.dataloader import test_dataset


# ==================== 全局配置 ====================
NET_NAME = 'UAD-Net'
METHOD_NAME = 'uadnet_full'

# 模型检查点路径
CHECKPOINT_PATH = Path('checkpoints') / NET_NAME / METHOD_NAME / 'best_model.pth'

# 数据集配置
DATASETS_ROOT = Path('D:/Datasets/Camouflaged_object_segmentation/TestDataset')
TEST_DATASETS = ['CHAMELEON', 'CAMO', 'COD10K', 'NC4K']

# 预测结果保存路径
PRED_ROOT = Path('results') / NET_NAME / METHOD_NAME


# ==================== 参数解析 ====================

def get_parser() -> argparse.Namespace:
    """
    解析命令行参数
    
    Returns:
        参数命名空间
    """
    parser = argparse.ArgumentParser(description='UAD-Net Testing Script')
    
    # 模型配置
    parser.add_argument('--net_name', type=str, default=NET_NAME, 
                        help='Network name')
    parser.add_argument('--method', type=str, default=METHOD_NAME, 
                        help='Method name')
    parser.add_argument('--checkpoint', type=str, default=str(CHECKPOINT_PATH), 
                        help='Model checkpoint path')
    
    # 测试配置
    parser.add_argument('--testsize', type=int, default=352, 
                        help='Test image size')
    parser.add_argument('--datasets_root', type=str, default=str(DATASETS_ROOT), 
                        help='Root path of test datasets')
    parser.add_argument('--test_datasets', nargs='+', default=TEST_DATASETS, 
                        help='List of test dataset names')
    parser.add_argument('--pred_root', type=str, default=str(PRED_ROOT), 
                        help='Root path for saving predictions')
    
    # 设备配置
    parser.add_argument('--device', type=str, default='cuda:0', 
                        help='Device (cuda:0 or cpu)')
    
    # 其他选项
    parser.add_argument('--save_intermediate', action='store_true', 
                        help='Save intermediate features (edge maps)')
    parser.add_argument('--batch_process', action='store_true', 
                        help='Process all datasets in batch')
    
    return parser.parse_args()


# ==================== 辅助函数 ====================

def save_test_config(opt: argparse.Namespace) -> None:
    """
    保存测试配置到文本文件
    
    Args:
        opt: 命令行参数
    """
    config_path = Path(opt.pred_root)
    config_path.mkdir(parents=True, exist_ok=True)
    
    config_file = config_path / f'{opt.net_name}_{opt.method}_test_config.txt'
    
    with open(config_file, 'a', encoding='utf-8') as f:
        f.write('='*60 + '\n')
        f.write(f'Test Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write('='*60 + '\n\n')
        
        f.write('Configuration:\n')
        f.write('-'*60 + '\n')
        for key, value in sorted(vars(opt).items()):
            f.write(f'{key:20s}: {value}\n')
        f.write('\n')
    
    print(f'✓ Test configuration saved to: {config_file}')


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """
    加载模型
    
    Args:
        checkpoint_path: 检查点路径
        device: 设备
        
    Returns:
        加载的模型
    """
    print(f'\nLoading model from: {checkpoint_path}')
    
    model = Net()
    
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print('✓ Model loaded successfully')
    
    return model


def normalize_prediction(pred: np.ndarray) -> np.ndarray:
    """
    归一化预测结果到 [0, 1]
    
    Args:
        pred: 预测结果
        
    Returns:
        归一化后的结果
    """
    pred_min = pred.min()
    pred_max = pred.max()
    return (pred - pred_min) / (pred_max - pred_min + 1e-8)


def _save_intermediate_results(
    pred_edge: torch.Tensor,
    gt_shape: Tuple[int, int],
    name: str,
    save_path: Path
) -> None:
    """
    保存中间特征结果
    
    Args:
        pred_edge: 边缘预测
        gt_shape: GT 形状
        name: 图像名称
        save_path: 保存路径
    """
    # 保存边缘结果
    edge_path = save_path / 'edge'
    edge_path.mkdir(exist_ok=True)
    
    # 调整尺寸
    pred_edge = F.interpolate(
        pred_edge, 
        size=gt_shape, 
        mode='bilinear', 
        align_corners=True
    )
    
    # 转换为 numpy 并归一化
    pred_edge = pred_edge.cpu().numpy().squeeze()
    pred_edge = normalize_prediction(pred_edge)
    
    # 保存
    imageio.imsave(edge_path / name, img_as_ubyte(pred_edge))


# ==================== 测试函数 ====================

def test_single_dataset(
    model: torch.nn.Module,
    dataset_name: str,
    opt: argparse.Namespace
) -> None:
    """
    在单个数据集上进行测试
    
    Args:
        model: 测试模型
        dataset_name: 数据集名称
        opt: 配置参数
    """
    print(f'\n{"="*60}')
    print(f'Testing on {dataset_name}...')
    print(f'{"="*60}')
    
    # 构建路径
    data_path = Path(opt.datasets_root) / dataset_name
    save_path = Path(opt.pred_root) / dataset_name
    save_path.mkdir(parents=True, exist_ok=True)
    
    image_root = data_path / 'Imgs'
    gt_root = data_path / 'GT'
    
    # 检查路径是否存在
    if not image_root.exists():
        print(f'⚠ Warning: Image root not found: {image_root}')
        return
    
    print(f'Image root: {image_root}')
    print(f'GT root: {gt_root}')
    print(f'Save path: {save_path}')
    
    # 加载数据
    test_loader = test_dataset(str(image_root), str(gt_root), opt.testsize)
    
    if test_loader.size == 0:
        print(f'⚠ Warning: No images found in {dataset_name}')
        return
    
    print(f'Total images: {test_loader.size}')
    
    # 测试循环
    test_bar = tqdm(range(test_loader.size), desc=f'Processing {dataset_name}')
    
    for i in test_bar:
        try:
            # 加载数据
            image, gt, name = test_loader.load_data()
            image = image.to(opt.device)
            
            # 推理
            with torch.no_grad():
                pred_out, pred_edge = model(image)
            
            # 后处理主输出
            pred_out = F.interpolate(
                pred_out, 
                size=gt.shape, 
                mode='bilinear', 
                align_corners=True
            )
            pred_out = pred_out.sigmoid().cpu().numpy().squeeze()
            pred_out = normalize_prediction(pred_out)
            
            # 保存主结果
            imageio.imsave(save_path / name, img_as_ubyte(pred_out))
            
            # 保存中间结果（可选）
            if opt.save_intermediate:
                _save_intermediate_results(pred_edge, gt.shape, name, save_path)
                
        except Exception as e:
            print(f'\n✗ Error processing {name}: {str(e)}')
            continue
    
    print(f'✓ {dataset_name} testing completed. Results saved to {save_path}\n')


# ==================== 主函数 ====================

def main():
    """主测试流程"""
    # 解析参数
    opt = get_parser()
    
    # 设置设备
    opt.device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {opt.device}')
    
    # 保存配置
    save_test_config(opt)
    
    # 加载模型
    try:
        model = load_model(opt.checkpoint, opt.device)
    except Exception as e:
        print(f'✗ Error loading model: {str(e)}')
        return
    
    # 测试所有数据集
    print(f'\n{"="*60}')
    print(f'Starting testing on {len(opt.test_datasets)} datasets')
    print(f'{"="*60}')
    
    start_time = datetime.now()
    
    for dataset_name in opt.test_datasets:
        test_single_dataset(model, dataset_name, opt)
    
    # 总结
    total_time = datetime.now() - start_time
    print(f'\n{"="*60}')
    print(f'All testing completed!')
    print(f'Total time: {total_time}')
    print(f'Results saved to: {opt.pred_root}')
    print(f'{"="*60}\n')


if __name__ == '__main__':
    main()
