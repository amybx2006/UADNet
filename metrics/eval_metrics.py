#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    : eval_metrics.py
@Time    : 2026/04/18 20:04:22
@Author  : Bai Xueqiong
@Email   : amybx2006@sina.com
@Description: 伪装目标检测评估脚本
"""

# Copyright (c) 2026 by 丁铖, 白雪琼, All Rights Reserved.
# Licensed under the MIT License
# ========================================
# Project: UAD-Net Evaluation
# Original evaluation code by Lart Pang
# GitHub: https://github.com/lartpang
# ========================================

import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import cv2
import xlsxwriter
from tqdm import tqdm
from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure

# ==================== 配置参数 ====================

# 模型配置
MODEL_NAME = 'UAD-Net'
METHOD_NAME = 'uadnet_full'
BACKBONE_NAME = 'ConvNeXt'

# 数据集配置
DATASETS = ['CHAMELEON', 'CAMO', 'COD10K', 'NC4K']

# 路径配置
MASK_ROOT = Path('D:/Datasets/Camouflaged_object_segmentation/TestDataset')
PRED_ROOT = Path('results') / MODEL_NAME / METHOD_NAME

# 输出配置
OUTPUT_DIR = Path('evaluation_results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================== 评估函数 ====================

def evaluate_single_dataset(
        dataset_name: str,
        mask_root: Path,
        pred_root: Path
) -> Dict[str, float]:
    """
    在单个数据集上评估模型性能

    Args:
        dataset_name: 数据集名称
        mask_root: Ground Truth 路径
        pred_root: 预测结果路径

    Returns:
        评估指标字典
    """
    print(f'\n{"=" * 70}')
    print(f'Evaluating on {dataset_name}...')
    print(f'{"=" * 70}')
    print(f'Mask root: {mask_root}')
    print(f'Pred root: {pred_root}')

    # 检查路径是否存在
    if not mask_root.exists():
        raise FileNotFoundError(f'Mask root not found: {mask_root}')
    if not pred_root.exists():
        raise FileNotFoundError(f'Prediction root not found: {pred_root}')

    # 初始化评估指标
    fm = Fmeasure()
    wfm = WeightedFmeasure()
    sm = Smeasure()
    em = Emeasure()
    mae = MAE()

    # 获取所有 mask 文件
    mask_files = sorted([f for f in mask_root.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.bmp']])

    if len(mask_files) == 0:
        raise ValueError(f'No mask files found in {mask_root}')

    print(f'Total images: {len(mask_files)}')

    # 遍历所有图像
    for mask_path in tqdm(mask_files, desc=f'Processing {dataset_name}'):
        mask_name = mask_path.name

        # 查找对应的预测文件
        pred_path = pred_root / mask_name
        if not pred_path.exists():
            # 尝试 .png 扩展名
            pred_path = pred_root / f'{mask_path.stem}.png'

        if not pred_path.exists():
            print(f'\n⚠ Warning: Prediction not found for {mask_name}')
            continue

        # 读取图像
        try:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)

            if mask is None:
                print(f'\n⚠ Warning: Failed to read mask: {mask_path}')
                continue
            if pred is None:
                print(f'\n⚠ Warning: Failed to read prediction: {pred_path}')
                continue

            # 更新指标
            fm.step(pred=pred, gt=mask)
            wfm.step(pred=pred, gt=mask)
            sm.step(pred=pred, gt=mask)
            em.step(pred=pred, gt=mask)
            mae.step(pred=pred, gt=mask)

        except Exception as e:
            print(f'\n✗ Error processing {mask_name}: {str(e)}')
            continue

    # 获取最终结果
    fm_results = fm.get_results()['fm']
    wfm_results = wfm.get_results()['wfm']
    sm_results = sm.get_results()['sm']
    em_results = em.get_results()['em']
    mae_results = mae.get_results()['mae']

    # 组织结果
    results = {
        'Smeasure': sm_results,
        'wFmeasure': wfm_results,
        'MAE': mae_results,
        'adpEm': em_results['adp'],
        'meanEm': em_results['curve'].mean(),
        'maxEm': em_results['curve'].max(),
        'adpFm': fm_results['adp'],
        'meanFm': fm_results['curve'].mean(),
        'maxFm': fm_results['curve'].max(),
    }

    # 打印结果
    print(f'\n📊 Results for {dataset_name}:')
    print(f'   S-measure:     {results["Smeasure"]:.4f}')
    print(f'   Max F-measure: {results["maxFm"]:.4f}')
    print(f'   Max E-measure: {results["maxEm"]:.4f}')
    print(f'   MAE:           {results["MAE"]:.4f}')

    return results


def save_results_to_txt(
        all_results: Dict[str, Dict[str, float]],
        output_path: Path,
        timestamp: str
):
    """
    保存评估结果到文本文件

    Args:
        all_results: 所有数据集的评估结果
        output_path: 输出文件路径
        timestamp: 时间戳
    """
    with open(output_path, 'a', encoding='utf-8') as f:
        f.write('=' * 70 + '\n')
        f.write(f'Evaluation Time: {timestamp}\n')
        f.write(f'Model: {MODEL_NAME}\n')
        f.write(f'Method: {METHOD_NAME}\n')
        f.write(f'Backbone: {BACKBONE_NAME}\n')
        f.write('=' * 70 + '\n\n')

        for dataset_name, results in all_results.items():
            f.write(f'{dataset_name}:\n')
            f.write('-' * 70 + '\n')
            for metric, value in results.items():
                f.write(f'  {metric:12s}: {value:.4f}\n')
            f.write('\n')

    print(f'\n✓ Results saved to: {output_path}')


def save_results_to_excel(
        all_results: Dict[str, Dict[str, float]],
        output_path: Path,
        timestamp: str
):
    """
    保存评估结果到 Excel 文件

    Args:
        all_results: 所有数据集的评估结果
        output_path: 输出文件路径
        timestamp: 时间戳
    """
    # 创建 Excel 文件
    workbook = xlsxwriter.Workbook(str(output_path))

    # 定义格式
    header_format = workbook.add_format({
        'bold': True,
        'align': 'center',
        'valign': 'vcenter',
        'bg_color': '#D7E4BD',
        'border': 1
    })

    metric_format = workbook.add_format({
        'align': 'center',
        'valign': 'vcenter',
        'border': 1
    })

    value_format = workbook.add_format({
        'align': 'center',
        'valign': 'vcenter',
        'border': 1,
        'num_format': '0.0000'
    })

    # ==================== 常用指标工作表 ====================
    common_sheet = workbook.add_worksheet('Common Metrics')

    # 设置列宽
    common_sheet.set_column('A:P', 12)

    # 常用指标列表
    common_metrics = ['Smeasure', 'maxFm', 'maxEm', 'MAE']

    # 写入数据集标题
    col = 0
    for dataset_name in DATASETS:
        common_sheet.merge_range(
            0, col, 0, col + len(common_metrics) - 1,
            dataset_name,
            header_format
        )

        # 写入指标名称
        for i, metric in enumerate(common_metrics):
            common_sheet.write(1, col + i, metric, metric_format)

        # 写入指标值
        if dataset_name in all_results:
            for i, metric in enumerate(common_metrics):
                value = all_results[dataset_name].get(metric, 0)
                common_sheet.write(2, col + i, value, value_format)

        col += len(common_metrics)

    # ==================== 所有指标工作表 ====================
    all_sheet = workbook.add_worksheet('All Metrics')

    # 设置列宽
    all_sheet.set_column('A:AJ', 12)

    # 所有指标列表
    all_metrics = list(next(iter(all_results.values())).keys())

    # 写入数据集标题
    col = 0
    for dataset_name in DATASETS:
        all_sheet.merge_range(
            0, col, 0, col + len(all_metrics) - 1,
            dataset_name,
            header_format
        )

        # 写入指标名称
        for i, metric in enumerate(all_metrics):
            all_sheet.write(1, col + i, metric, metric_format)

        # 写入指标值
        if dataset_name in all_results:
            for i, metric in enumerate(all_metrics):
                value = all_results[dataset_name].get(metric, 0)
                all_sheet.write(2, col + i, value, value_format)

        col += len(all_metrics)

    # ==================== 信息工作表 ====================
    info_sheet = workbook.add_worksheet('Information')
    info_sheet.set_column('A:B', 20)

    info_data = [
        ['Evaluation Time', timestamp],
        ['Model', MODEL_NAME],
        ['Method', METHOD_NAME],
        ['Backbone', BACKBONE_NAME],
        ['Datasets', ', '.join(DATASETS)],
    ]

    for i, (key, value) in enumerate(info_data):
        info_sheet.write(i, 0, key, metric_format)
        info_sheet.write(i, 1, value, value_format)

    # 关闭文件
    workbook.close()

    print(f'✓ Excel file saved to: {output_path}')


# ==================== 主函数 ====================

def main():
    """主评估流程"""
    print('=' * 70)
    print('UAD-Net Evaluation Script')
    print('=' * 70)

    # 生成时间戳
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # 存储所有结果
    all_results = {}

    # 评估每个数据集
    for dataset_name in DATASETS:
        try:
            # 构建路径
            mask_root = MASK_ROOT / dataset_name / 'GT'
            pred_root = PRED_ROOT / dataset_name

            # 评估
            results = evaluate_single_dataset(dataset_name, mask_root, pred_root)
            all_results[dataset_name] = results

        except Exception as e:
            print(f'\n✗ Error evaluating {dataset_name}: {str(e)}')
            continue

    # 检查是否有结果
    if not all_results:
        print('\n✗ No results to save. Exiting...')
        return

    # 保存结果到文本文件
    txt_path = OUTPUT_DIR / f'{MODEL_NAME}_{METHOD_NAME}_results_{timestamp}.txt'
    save_results_to_txt(all_results, txt_path, timestamp)

    # 保存结果到 Excel 文件
    excel_path = OUTPUT_DIR / f'{MODEL_NAME}_{METHOD_NAME}_metrics_{timestamp}.xlsx'
    save_results_to_excel(all_results, excel_path, timestamp)

    # 打印总结
    print('\n' + '=' * 70)
    print('Evaluation Summary')
    print('=' * 70)

    # 计算平均指标
    avg_metrics = {}
    for metric in ['Smeasure', 'maxFm', 'maxEm', 'MAE']:
        values = [results[metric] for results in all_results.values()]
        avg_metrics[metric] = sum(values) / len(values)

    print(f'\nAverage across {len(all_results)} datasets:')
    print(f'  S-measure:     {avg_metrics["Smeasure"]:.4f}')
    print(f'  Max F-measure: {avg_metrics["maxFm"]:.4f}')
    print(f'  Max E-measure: {avg_metrics["maxEm"]:.4f}')
    print(f'  MAE:           {avg_metrics["MAE"]:.4f}')

    print('\n' + '=' * 70)
    print('Evaluation completed successfully! ✓')
    print('=' * 70)


if __name__ == '__main__':
    main()
