# -*- coding: utf-8 -*-
r"""
==============================================================================
[已废弃 / DEPRECATED] 请勿用于新的标注流程
==============================================================================

标注工具已从 labelme 改为 Label Studio，数据集改由
E:\Work\LabelStudio\export-yolo.ps1 直接导出，本脚本不再参与主流程。

新流程：
    1. 在 Label Studio 中标注（项目 4：DNF-YOLO验证｜目标检测）
    2. 导出：
         E:\Work\LabelStudio\export-yolo.ps1 -Project 4 -SplitByGroup
       产物直接落到 D:\Project\yolov5-DNF\datasets\dnf，
       并同时生成 classes.txt 与 data.yaml（类别顺序取自 data/dnf.yaml）。

保留本脚本的原因：
    - 兼容历史上用 labelme 标好的 xxx.jpg + xxx.json 数据
    - 仍有旧数据需要转换时可以继续使用

退出条件（满足其一即可删除本文件）：
    - 确认没有 labelme 格式的旧标注数据需要转换
    - 或已完成全部旧数据向 Label Studio 的迁移

------------------------------------------------------------------------------
以下为原始说明（labelme 流程）：

数据集准备脚本：labelme 标注 → ultralytics YOLO 训练数据集。

整合了原 json2yolo.py（labelme JSON → YOLO txt）与 datasets_utils.py（拷贝+划分）的功能，
直接输出 ultralytics 标准目录结构：

    <dst>/
      images/train/*.jpg    images/val/*.jpg
      labels/train/*.txt    labels/val/*.txt

输入目录里每个样本是「同名」的一对文件：xxx.jpg + xxx.json（labelme 矩形标注）。

用法（旧流程）：
    python prepare_dataset.py --src path/to/labeled --dst datasets/dnf --val-ratio 0.1
"""

import argparse
import json
import os
import random
import shutil

# 9 类映射（沿用原项目，与 data/dnf.yaml 一致）
NAME2ID = {'hero': 0, 'small_map': 1, 'monster': 2, 'money': 3, 'material': 4,
           'door': 5, 'BOSS': 6, 'box': 7, 'options': 8}


def convert_labelme_to_yolo(json_path):
    """把单个 labelme JSON 转成 YOLO txt 内容字符串（仅矩形框）。"""
    with open(json_path, 'r', encoding='gb2312') as f:
        data = json.load(f)

    img_w = data['imageWidth']
    img_h = data['imageHeight']
    lines = []

    for shape in data['shapes']:
        label = shape['label']
        if label not in NAME2ID:
            print(f'  [跳过] 未知类别 {label!r} (文件 {os.path.basename(json_path)})')
            continue
        if shape.get('shape_type') != 'rectangle':
            continue

        (x1, y1), (x2, y2) = shape['points'][0], shape['points'][1]
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])

        cx = (x1 + x2) / 2.0 / img_w
        cy = (y1 + y2) / 2.0 / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        lines.append(f'{NAME2ID[label]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}')

    return '\n'.join(lines)


def main():
    import sys
    print('', file=sys.stderr)
    print('!' * 74, file=sys.stderr)
    print('[警告] 本脚本已废弃，请勿用于新的标注流程。', file=sys.stderr)
    print('       新流程请用 Label Studio 标注后执行：', file=sys.stderr)
    print(r'         E:\Work\LabelStudio\export-yolo.ps1 -Project 4 -SplitByGroup', file=sys.stderr)
    print('       仅在转换历史 labelme 数据时才应使用本脚本。', file=sys.stderr)
    print('!' * 74, file=sys.stderr)
    print('', file=sys.stderr)

    ap = argparse.ArgumentParser(description='[已废弃] labelme → YOLO 数据集')
    ap.add_argument('--src', required=True, help='labelme 标注目录（含同名 jpg+json）')
    ap.add_argument('--dst', default='datasets/dnf', help='输出数据集根目录')
    ap.add_argument('--val-ratio', type=float, default=0.1, help='验证集比例')
    ap.add_argument('--seed', type=int, default=42, help='随机划分种子')
    args = ap.parse_args()

    random.seed(args.seed)

    # 收集样本（jpg 且有同名 json）
    samples = []
    for fn in os.listdir(args.src):
        if fn.lower().endswith('.jpg'):
            stem = os.path.splitext(fn)[0]
            json_path = os.path.join(args.src, stem + '.json')
            if os.path.isfile(json_path):
                samples.append((os.path.join(args.src, fn), json_path, stem))
    if not samples:
        print(f'错误：{args.src} 下没有找到成对的 jpg+json 样本。')
        return

    random.shuffle(samples)
    n_val = max(1, int(len(samples) * args.val_ratio))
    val_set = set(samples[:n_val])

    for sub in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        os.makedirs(os.path.join(args.dst, sub), exist_ok=True)

    for img_path, json_path, stem in samples:
        split = 'val' if (img_path, json_path, stem) in val_set else 'train'
        shutil.copy(img_path, os.path.join(args.dst, 'images', split, stem + '.jpg'))
        yolo_txt = convert_labelme_to_yolo(json_path)
        if not yolo_txt:
            print(f'  [警告] {stem} 没有有效标注，跳过 label（图片仍已拷贝）')
            continue
        with open(os.path.join(args.dst, 'labels', split, stem + '.txt'), 'w') as f:
            f.write(yolo_txt + '\n')

    print(f'完成：{len(samples)} 个样本，训练 {len(samples) - n_val}，验证 {n_val}')
    print(f'输出目录：{os.path.abspath(args.dst)}')
    print()
    print('[注意] 本脚本已废弃。新标注流程：')
    print(r'  E:\Work\LabelStudio\export-yolo.ps1 -Project 4 -SplitByGroup')
    print(r'  产物直接输出到 D:\Project\yolov5-DNF\datasets\dnf，无需再改 dnf.yaml 的 path')


if __name__ == '__main__':
    main()

