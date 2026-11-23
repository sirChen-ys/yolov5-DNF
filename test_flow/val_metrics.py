# -*- coding: utf-8 -*-
"""用训练好的 best.pt 在验证集上跑一遍官方评估，确认权重可用。

用法（在项目根目录执行）：
    python test_flow/val_metrics.py
    python test_flow/val_metrics.py --data data/dnf.yaml --imgsz 640

注意：per-class 指标直接取 `metrics.summary()`，
不能按 class_id 去索引 box.ap50/box.ap —— ultralytics 会把「验证集里 0 实例」
的类别从这些数组里剔除（本数据集里是 BOSS），按下标取会整体错位。
"""

import argparse
import os
import sys

import torch
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser(description='best.pt 验证集评估')
    p.add_argument('--weights', default='best.pt')
    p.add_argument('--data', default='datasets/dnf/data.yaml')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--conf', type=float, default=0.001, help='val 的 conf 阈值')
    p.add_argument('--iou', type=float, default=0.7)
    p.add_argument('--device', default='', help='0 / cpu，留空自动')
    p.add_argument('--workers', type=int, default=0, help='DataLoader 线程（Windows 上 0 最稳）')
    p.add_argument('--name', default='val', help='runs/test_flow/<name>')
    return p.parse_args()


def main():
    args = parse_args()
    os.chdir(ROOT)

    device = args.device or ('0' if torch.cuda.is_available() else 'cpu')
    print('=' * 72)
    print('权重   :', args.weights)
    print('数据   :', args.data)
    print('设备   :', device, '| CUDA:', torch.cuda.is_available())
    print('=' * 72)

    model = YOLO(args.weights)
    print('模型类别   :', model.names)
    print('任务/参数量:', model.task, sum(p.numel() for p in model.model.parameters()))

    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=device,
        workers=args.workers,
        project='runs/test_flow',
        name=args.name,
        exist_ok=True,
        plots=False,
        verbose=False,
    )

    box = metrics.box
    print('\n---- 总体指标 ----')
    print('mAP50    : %.4f' % box.map50)
    print('mAP50-95 : %.4f' % box.map)
    print('mean P   : %.4f' % box.mp)
    print('mean R   : %.4f' % box.mr)
    if getattr(metrics, 'speed', None):
        print('speed(ms):', {k: round(v, 2) for k, v in metrics.speed.items()})

    print('\n---- 每类指标（来自 summary()）----')
    print('%-11s %6s %9s %7s %7s %7s %9s' % ('class', 'imgs', 'instances', 'P', 'R', 'mAP50', 'mAP50-95'))
    rows = metrics.summary()
    covered = set()
    for row in rows:
        name = row['Class']
        covered.add(name)
        print('%-11s %6d %9d %7.3f %7.3f %7.3f %9.3f' % (
            name, row['Images'], row['Instances'],
            float(row['Box-P']), float(row['Box-R']),
            float(row['mAP50']), float(row['mAP50-95'])))

    missing = [n for i, n in metrics.names.items() if n not in covered]
    if missing:
        print('\n[注意] 验证集里没有实例、无法评估的类别: %s' % ', '.join(missing))

    print('\n评估结果已保存到: runs/test_flow/%s' % args.name)
    return 0


if __name__ == '__main__':
    sys.exit(main())
