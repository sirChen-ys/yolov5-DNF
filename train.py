# -*- coding: utf-8 -*-
"""
DNF YOLOv11 训练脚本（基于 ultralytics）。

训练完成后，最优权重在 runs/detect/<name>/weights/best.pt，
把它拷贝到项目根目录作为推理权重（main2.py / yolo11_detect.py 默认加载 best.pt）。

示例：
    # 从头训练（推荐用 yolo11n 起步，速度快；显存/精度要求高可用 yolo11s/m）
    python train.py --weights yolo11n.pt --epochs 150 --imgsz 640 --batch 16 --device 0

    # 在已有权重上继续训练
    python train.py --weights best.pt --epochs 50 --device 0

    # 无 GPU
    python train.py --weights yolo11n.pt --epochs 50 --device cpu --batch 8
"""

import argparse

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description='DNF YOLOv11 训练')
    p.add_argument('--weights', default='yolo11n.pt',
                   help='预训练权重 (yolo11n/s/m/l/x.pt 或已有 best.pt)')
    p.add_argument('--data', default='data/dnf.yaml',
                   help='数据集配置文件路径')
    p.add_argument('--epochs', type=int, default=150, help='训练轮数')
    p.add_argument('--imgsz', type=int, default=640, help='输入图像尺寸')
    p.add_argument('--batch', type=int, default=16, help='batch size（显存不足就调小）')
    p.add_argument('--device', default='', help='训练设备，0 表示 GPU0，cpu 表示 CPU，留空自动')
    p.add_argument('--name', default='dnf', help='输出目录名 runs/detect/<name>')
    p.add_argument('--patience', type=int, default=50, help='早停耐心（epoch 数无提升则停止）')
    p.add_argument('--workers', type=int, default=8, help='数据加载线程数')
    p.add_argument('--lr0', type=float, default=0.01, help='初始学习率')
    p.add_argument('--resume', action='store_true', help='从上次中断处继续训练')
    p.add_argument('--pretrained', action='store_true',
                   help='是否加载预训练权重（默认 True，仅当 --weights 为空时可控）')
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device if args.device else None

    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        lr0=args.lr0,
        resume=args.resume,
        pretrained=args.pretrained,
        project='runs/detect',
        exist_ok=True,
    )
    print('\n训练完成。最优权重：runs/detect/%s/weights/best.pt' % args.name)
    print('请将其拷贝到项目根目录 best.pt 供推理脚本使用。')


if __name__ == '__main__':
    main()
