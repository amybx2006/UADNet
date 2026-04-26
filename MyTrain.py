#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    : MyTrain.py
@Time    : 2026/04/26 11:56:00
@Author  : Bai Xueqiong
@Email   : amybx2006@sina.com
@Description: UAD-Net 训练脚本
"""

# Copyright (c) 2026 Bai Xueqiong, All Rights Reserved.
# Licensed under the MIT License
# ========================================

import os
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.UADNet import Net
from utils.dataloader_edge import get_loader, test_dataset_new
from utils.utils import clip_gradient, AvgMeter
from py_sod_metrics import Emeasure, Fmeasure, Smeasure

# ==================== 全局配置 ====================
NET_NAME = 'UAD-Net'
METHOD_NAME = 'uadnet_full'  #

# 数据集路径
TRAIN_DATASET_PATH = r'D:\Camouflaged_object_segmentation\Datasets\TrainDataset'
EVAL_DATASET_PATH = r'D:\Camouflaged_object_segmentation\Datasets\TestDataset\CAMO'

# 时间戳
TIME_NOW = datetime.now().strftime('%Y%m%d_%H%M')

# 路径配置
LOG_PATH = Path('log') / NET_NAME / TIME_NOW
TRAIN_SAVE_PATH = Path('checkpoints') / NET_NAME / TIME_NOW

# 继续训练配置
LOAD_CHECKPOINT = False
LAST_LOG_PATH = Path('log') / NET_NAME / '20231127_1113'
EPOCH_START_LAST = 26
CHECKPOINT_MODEL_PATH = Path('checkpoints') / NET_NAME / 'model_last.pth'
CHECKPOINT_OPTIMIZER_PATH = Path('checkpoints') / NET_NAME / 'optimizer_last.pth'

# 超参数常量
POOL_KERNEL_SIZE = 31
POOL_PADDING = 15
LOSS_WEIGHT_EDGE = 3
LOG_INTERVAL = 20
SAVE_INTERVAL = 10


# ==================== 损失函数 ====================

def structure_loss(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    结构损失函数：加权 BCE + 加权 IoU

    Args:
        pred: 预测结果 (B, 1, H, W)
        mask: 真实标签 (B, 1, H, W)

    Returns:
        损失值
    """
    # 计算权重：边缘区域权重更高
    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=POOL_KERNEL_SIZE, stride=1, padding=POOL_PADDING) - mask
    )

    # 加权二值交叉熵
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    # 加权 IoU
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def dice_loss(predict: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Dice 损失函数

    Args:
        predict: 预测结果
        target: 真实标签

    Returns:
        Dice 损失值
    """
    smooth = 1.0
    p = 2

    # 展平张量
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)

    # 计算 Dice 系数
    num = torch.sum(predict * target, dim=1) * 2 + smooth
    den = torch.sum(predict.pow(p) + target.pow(p), dim=1) + smooth
    loss = 1 - num / den

    return loss.mean()


# ==================== 训练函数 ====================

def train(
        train_loader,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        opt: argparse.Namespace,
        writer: SummaryWriter,
        start_time: datetime
) -> None:
    """
    训练一个 epoch

    Args:
        train_loader: 训练数据加载器
        model: 模型
        optimizer: 优化器
        epoch: 当前 epoch
        opt: 配置参数
        writer: TensorBoard writer
        start_time: 训练开始时间
    """
    model.train()

    # 创建保存目录
    save_path = Path(opt.train_save)
    save_path.mkdir(parents=True, exist_ok=True)

    # 损失记录器
    loss_meters = {
        'out': AvgMeter(),
        'o1': AvgMeter(),
        'o2': AvgMeter(),
        'edge': AvgMeter(),
        'total': AvgMeter()
    }

    # 训练循环
    train_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{opt.epoch}')

    try:
        for i, (images, gts, edges) in enumerate(train_bar, start=1):
            # 数据迁移到 GPU
            images = images.to(opt.device)
            gts = gts.to(opt.device)
            edges = edges.to(opt.device)

            # 前向传播
            optimizer.zero_grad()
            pred_out, pred_o1, pred_o2, pred_edge = model(images)

            # 计算损失
            loss_out = structure_loss(pred_out, gts)
            loss_o1 = structure_loss(pred_o1, gts)
            loss_o2 = structure_loss(pred_o2, gts)
            loss_edge = dice_loss(pred_edge, edges)

            total_loss = loss_out + loss_o1 + loss_o2 + LOSS_WEIGHT_EDGE * loss_edge

            # 反向传播
            total_loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            # 记录损失
            batch_size = opt.batchsize
            loss_meters['out'].update(loss_out.item(), batch_size)
            loss_meters['o1'].update(loss_o1.item(), batch_size)
            loss_meters['o2'].update(loss_o2.item(), batch_size)
            loss_meters['edge'].update(loss_edge.item(), batch_size)
            loss_meters['total'].update(total_loss.item(), batch_size)

            # 日志记录
            if i % LOG_INTERVAL == 0 or i == len(train_loader):
                logging.info(
                    f'[Train] Epoch [{epoch:03d}/{opt.epoch:03d}], '
                    f'Step [{i:04d}/{len(train_loader):04d}], '
                    f'Loss: {loss_meters["total"].show():.4f}'
                )

            train_bar.set_postfix({'loss': f'{total_loss.item():.5f}'})

        # Epoch 结束统计
        time_cost = datetime.now() - start_time
        logging.info(
            f'[Train] Epoch {epoch} finished. Time cost: {time_cost}, '
            f'Avg Loss: {loss_meters["total"].show():.4f}'
        )

        # TensorBoard 记录
        for name, meter in loss_meters.items():
            writer.add_scalar(f'loss/{name}', meter.show(), epoch)

        # 保存模型
        _save_checkpoint(model, optimizer, epoch, opt, loss_meters['total'].show())

    except KeyboardInterrupt:
        logging.warning('Training interrupted by user')
        torch.save(model.state_dict(), save_path / f'interrupted_epoch_{epoch}.pth')
        raise


def _save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        opt: argparse.Namespace,
        current_loss: float
) -> None:
    """
    保存模型检查点

    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前 epoch
        opt: 配置参数
        current_loss: 当前损失
    """
    save_path = Path(opt.train_save)

    # 保存最新模型
    torch.save(model.state_dict(), save_path / f'{opt.net_name}_last.pth')
    torch.save(optimizer.state_dict(), save_path / f'{opt.net_name}_last_optimizer.pth')
    logging.info(f'[Save] Latest checkpoint saved to {save_path}')

    # 保存最佳模型
    if not hasattr(opt, 'best_loss'):
        opt.best_loss = current_loss

    if current_loss < opt.best_loss:
        opt.best_loss = current_loss
        torch.save(model.state_dict(), save_path / f'{opt.net_name}_best.pth')
        logging.info(f'[Save] Best checkpoint saved with loss: {opt.best_loss:.4f}')

    # 定期保存
    if (epoch + 1) % SAVE_INTERVAL == 0:
        torch.save(model.state_dict(), save_path / f'{opt.net_name}_epoch_{epoch}.pth')
        torch.save(optimizer.state_dict(), save_path / f'{opt.net_name}_epoch_{epoch}_optimizer.pth')
        logging.info(f'[Save] Epoch {epoch} checkpoint saved')


# ==================== 评估函数 ====================

def evaluate(
        test_loader,
        model: torch.nn.Module,
        epoch: int,
        opt: argparse.Namespace,
        writer: SummaryWriter
) -> Dict[str, float]:
    """
    评估模型性能

    Args:
        test_loader: 测试数据加载器
        model: 模型
        epoch: 当前 epoch
        opt: 配置参数
        writer: TensorBoard writer

    Returns:
        评估指标字典
    """
    model.eval()

    # 初始化评估指标
    fm = Fmeasure()
    sm = Smeasure()
    em = Emeasure()

    with torch.no_grad():
        test_bar = tqdm(range(test_loader.size), desc=f'Eval Epoch {epoch}')

        for i in test_bar:
            # 加载数据
            images, gts, _, _ = test_loader.load_data()
            images = images.to(opt.device)

            # 前向传播
            pred_out, _, _, _ = model(images)

            # 调整预测尺寸
            pred_out = F.interpolate(
                pred_out,
                size=gts.shape,
                mode='bilinear',
                align_corners=True
            )

            # 归一化预测结果
            pred_out = pred_out.sigmoid().cpu().numpy().squeeze()
            pred_out = (pred_out - pred_out.min()) / (pred_out.max() - pred_out.min() + 1e-8)

            # 更新指标
            fm.step(pred=pred_out, gt=gts)
            sm.step(pred=pred_out, gt=gts)
            em.step(pred=pred_out, gt=gts)

    # 获取评估结果
    metrics = {
        'Sm': sm.get_results()['sm'],
        'maxFm': fm.get_results()['fm']['curve'].max().round(3),
        'maxEm': em.get_results()['em']['curve'].max().round(3)
    }
    metrics['score'] = metrics['Sm'] + metrics['maxFm'] + metrics['maxEm']

    # 更新最佳结果
    if not hasattr(opt, 'best_score'):
        opt.best_score = 0
        opt.best_epoch = 0
        opt.best_metrics = {}

    if metrics['score'] > opt.best_score:
        opt.best_score = metrics['score']
        opt.best_epoch = epoch
        opt.best_metrics = metrics

        # 保存最佳模型
        save_path = Path(opt.train_save) / f'{opt.net_name}_best_eval_epoch_{epoch}.pth'
        torch.save(model.state_dict(), save_path)
        logging.info(f'[Eval] New best model saved at epoch {epoch}')

    # 日志记录
    logging.info(
        f'[Eval] Epoch {epoch}: Sm={metrics["Sm"]:.3f}, '
        f'maxFm={metrics["maxFm"]:.3f}, maxEm={metrics["maxEm"]:.3f}, '
        f'Score={metrics["score"]:.3f}'
    )

    if opt.best_epoch > 0:
        logging.info(
            f'[Eval] Best Epoch {opt.best_epoch}: '
            f'Sm={opt.best_metrics["Sm"]:.3f}, '
            f'maxFm={opt.best_metrics["maxFm"]:.3f}, '
            f'maxEm={opt.best_metrics["maxEm"]:.3f}, '
            f'Score={opt.best_score:.3f}'
        )

    # TensorBoard 记录
    writer.add_scalar('eval/Sm', metrics['Sm'], epoch)
    writer.add_scalar('eval/maxFm', metrics['maxFm'], epoch)
    writer.add_scalar('eval/maxEm', metrics['maxEm'], epoch)
    writer.add_scalar('eval/score', metrics['score'], epoch)

    return metrics


# ==================== 参数解析 ====================

def get_parser() -> argparse.Namespace:
    """
    解析命令行参数

    Returns:
        参数命名空间
    """
    parser = argparse.ArgumentParser(description='UAD-Net Training Script')

    # 模型配置
    parser.add_argument('--net_name', type=str, default=NET_NAME, help='Network name')
    parser.add_argument('--method', type=str, default=METHOD_NAME, help='Method name')

    # 训练配置
    parser.add_argument('--epoch', type=int, default=35, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batchsize', type=int, default=2, help='Batch size')
    parser.add_argument('--trainsize', type=int, default=352, help='Training image size')
    parser.add_argument('--clip', type=float, default=0.5, help='Gradient clipping margin')

    # 路径配置
    parser.add_argument('--train_path', type=str, default=TRAIN_DATASET_PATH, help='Train dataset path')
    parser.add_argument('--eval_path', type=str, default=EVAL_DATASET_PATH, help='Eval dataset path')
    parser.add_argument('--train_save', type=str, default=str(TRAIN_SAVE_PATH), help='Checkpoint save path')
    parser.add_argument('--log_path', type=str, default=str(LOG_PATH), help='Log save path')

    # 检查点配置
    parser.add_argument('--load_checkpoint', action='store_true', default=LOAD_CHECKPOINT, help='Load checkpoint')
    parser.add_argument('--checkpoint_model', type=str, default=str(CHECKPOINT_MODEL_PATH),
                        help='Model checkpoint path')
    parser.add_argument('--checkpoint_optimizer', type=str, default=str(CHECKPOINT_OPTIMIZER_PATH),
                        help='Optimizer checkpoint path')
    parser.add_argument('--epoch_start', type=int, default=0, help='Start epoch')

    # 设备配置
    parser.add_argument('--device', type=str, default='cuda:0', help='Device (cuda:0 or cpu)')

    return parser.parse_args()


# ==================== 主函数 ====================

def main():
    """主训练流程"""
    # 解析参数
    opt = get_parser()

    # 设置设备
    opt.device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {opt.device}')

    # 创建目录
    Path(opt.train_save).mkdir(parents=True, exist_ok=True)
    Path(opt.log_path).mkdir(parents=True, exist_ok=True)

    # 配置日志
    logging.basicConfig(
        filename=Path(opt.log_path) / 'train.log',
        format='[%(asctime)s-%(filename)s-%(levelname)s] %(message)s',
        level=logging.INFO,
        filemode='a',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info('=' * 50)
    logging.info('Training started')
    logging.info(f'Config: {vars(opt)}')

    # 构建模型
    model = Net().to(opt.device)
    logging.info(f'Model: {opt.net_name}')

    # 优化器和学习率调度器
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=20, eta_min=1e-5
    )

    # 加载检查点
    if opt.load_checkpoint:
        if Path(opt.checkpoint_model).exists():
            model.load_state_dict(torch.load(opt.checkpoint_model, map_location=opt.device))
            optimizer.load_state_dict(torch.load(opt.checkpoint_optimizer, map_location=opt.device))
            opt.epoch_start = EPOCH_START_LAST
            logging.info(f'Checkpoint loaded from {opt.checkpoint_model}')
        else:
            logging.warning(f'Checkpoint not found: {opt.checkpoint_model}')

    # 数据加载器
    train_loader = get_loader(
        image_root=Path(opt.train_path) / 'Imgs',
        gt_root=Path(opt.train_path) / 'gt',
        edge_root=Path(opt.train_path) / 'edge',
        batchsize=opt.batchsize,
        trainsize=opt.trainsize
    )

    test_loader = test_dataset_new(
        image_root=Path(opt.eval_path) / 'Imgs',
        gt_root=Path(opt.eval_path) / 'gt',
        testsize=opt.trainsize
    )

    logging.info(f'Train samples: {len(train_loader.dataset)}')
    logging.info(f'Test samples: {test_loader.size}')

    # TensorBoard
    train_writer = SummaryWriter(Path(opt.log_path) / 'train')
    eval_writer = SummaryWriter(Path(opt.log_path) / 'eval')

    # 训练循环
    start_time = datetime.now()
    logging.info(f'Training started at {start_time}')

    for epoch in range(opt.epoch_start + 1, opt.epoch + 1):
        # 训练
        train(train_loader, model, optimizer, epoch, opt, train_writer, start_time)

        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        train_writer.add_scalar('lr', current_lr, epoch)
        logging.info(f'Learning rate: {current_lr:.6f}')

        # 评估
        evaluate(test_loader, model, epoch, opt, eval_writer)

    # 训练结束
    total_time = datetime.now() - start_time
    logging.info(f'Training finished. Total time: {total_time}')
    logging.info('=' * 50)

    train_writer.close()
    eval_writer.close()


if __name__ == '__main__':
    main()
