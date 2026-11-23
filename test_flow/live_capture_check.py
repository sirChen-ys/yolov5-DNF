# -*- coding: utf-8 -*-
"""实时抓屏 → 检测 的连通性检查（只截图推理，不发送任何按键、不弹窗）。

用来验证真正的 grabscreen.grab_screen + YoloDetector 这一环能不能跑通。
抓的是当前桌面（不带 DNF 也能跑，只是大概率检测不到目标）。

用法（在项目根目录执行）：
    python test_flow/live_capture_check.py
    python test_flow/live_capture_check.py --frames 5
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser(description='实时抓屏+检测连通性检查')
    ap.add_argument('--frames', type=int, default=5)
    ap.add_argument('--weights', default='best.pt')
    ap.add_argument('--region', default='0,0,1280,800', help='left,top,right,bottom')
    ap.add_argument('--conf', type=float, default=0.3)
    args = ap.parse_args()

    os.chdir(ROOT)
    left, top, right, bottom = (int(x) for x in args.region.split(','))
    region = (left, top, right, bottom)

    from grabscreen import grab_screen
    from yolo_detector import YoloDetector, NAMES

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    det = YoloDetector(args.weights, device=device, imgsz=640,
                       conf_thres=args.conf, iou_thres=0.2, names=NAMES)

    out_dir = os.path.join(ROOT, 'runs', 'test_flow', 'live')
    os.makedirs(out_dir, exist_ok=True)

    times = []
    for i in range(args.frames):
        t0 = time.perf_counter()
        img0 = grab_screen(region)
        t_grab = time.perf_counter()
        if img0 is None or not isinstance(img0, np.ndarray) or img0.ndim != 3 or img0.shape[2] != 4:
            print('[FAILED] grab_screen 返回异常:', type(img0), getattr(img0, 'shape', None))
            return 1
        img = cv2.cvtColor(img0, cv2.COLOR_BGRA2BGR)
        xywh, cls_ids, confs, xyxy = det.detect(img)
        t1 = time.perf_counter()
        times.append((t_grab - t0, t1 - t_grab))
        print('第 %d 帧: 抓屏 %dx%d, 耗时 抓屏 %.0f ms + 推理 %.0f ms, 目标 %d 个'
              % (i + 1, img.shape[1], img.shape[0],
                 (t_grab - t0) * 1000, (t1 - t_grab) * 1000, len(xywh)))
        for box, cid, conf in zip(xyxy, cls_ids, confs):
            print('    %-10s conf=%.2f box=(%d,%d,%d,%d)'
                  % (NAMES[cid], conf, box[0], box[1], box[2], box[3]))
        if i == 0:
            for box, cid, conf in zip(xyxy, cls_ids, confs):
                cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                              (0, 255, 0), 2)
                cv2.putText(img, '%s %.2f' % (NAMES[cid], conf),
                            (int(box[0]), max(int(box[1]) - 4, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            p = os.path.join(out_dir, 'live_frame.png')
            cv2.imwrite(p, img)
            print('   首帧已存:', p)

    g = np.mean([t[0] for t in times]) * 1000
    d = np.mean([t[1] for t in times]) * 1000
    print('\n平均: 抓屏 %.1f ms + 检测 %.1f ms = %.1f ms/帧 (约 %.1f FPS)'
          % (g, d, g + d, 1000.0 / (g + d)))
    print('[OK] 实时抓屏 + 检测链路可用')
    return 0


if __name__ == '__main__':
    sys.exit(main())
