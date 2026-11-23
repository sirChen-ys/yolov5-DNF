# -*- coding: utf-8 -*-
"""对准游戏窗口客户区抓屏 + 检测，检查实机识别效果。

不发送任何按键、不弹窗，只截图推理并保存带框图。

用法（在项目根目录执行）：
    python test_flow/probe_game_window.py
    python test_flow/probe_game_window.py --frames 10 --conf 0.3
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

GAME_CLASS = 'Dungeon & Fighter'
GAME_TITLE = '地下城与勇士'


def find_game_client_rect():
    """返回游戏客户区在屏幕上的 (left, top, right, bottom)；找不到返回 (None, None)。

    直接复用 game_window 的逻辑（含 DPI 感知），保证和 yolo11_detect / main2 一致。
    """
    from game_window import describe_region, find_game_window, get_game_region

    hwnd = find_game_window()
    if not hwnd:
        return None, None
    region = get_game_region()
    print('客户区在屏幕 :', describe_region(region))
    return region, hwnd


def main():
    ap = argparse.ArgumentParser(description='对准游戏窗口抓屏 + 检测')
    ap.add_argument('--frames', type=int, default=5)
    ap.add_argument('--weights', default='best.pt')
    ap.add_argument('--conf', type=float, default=0.3)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--activate', action='store_true',
                    help='先把游戏窗口切到前台再抓屏（grab_screen 抓的是桌面可见像素，被遮挡就抓不到游戏）')
    ap.add_argument('--resize', default='',
                    help='把抓到的画面缩放到该尺寸再推理，例如 1332x750（对齐训练数据）')
    args = ap.parse_args()

    os.chdir(ROOT)

    region, hwnd = find_game_client_rect()
    if region is None:
        print('[FAILED] 没找到游戏窗口 (class=%r / title=%r)'
              % (GAME_CLASS, GAME_TITLE))
        return 1
    # region 是 grab_screen 的口径：left, top, x2, y2（含末像素）
    print('游戏窗口句柄 : %s' % (hex(hwnd) if hwnd else ''))

    from grabscreen import grab_screen
    from yolo_detector import YoloDetector, NAMES

    if args.activate and hwnd:
        import win32con
        import win32gui
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        try:
            win32gui.SetForegroundWindow(hwnd)
        except Exception as exc:
            print('[warn] 切前台失败（Windows 会限制非前台进程抢焦点）:', exc)
        time.sleep(1.5)
        print('已尝试把游戏窗口切到前台')

    resize_to = None
    if args.resize:
        rw, rh = (int(x) for x in args.resize.lower().split('x'))
        resize_to = (rw, rh)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    det = YoloDetector(args.weights, device=device, imgsz=args.imgsz,
                       conf_thres=args.conf, iou_thres=0.2, names=NAMES)

    out_dir = os.path.join(ROOT, 'runs', 'test_flow', 'game')
    os.makedirs(out_dir, exist_ok=True)

    times = []
    last_img = None
    last_det = None
    for i in range(args.frames):
        t0 = time.perf_counter()
        img0 = grab_screen(region)
        t_grab = time.perf_counter()
        img = cv2.cvtColor(img0, cv2.COLOR_BGRA2BGR)
        if resize_to:
            img = cv2.resize(img, resize_to, interpolation=cv2.INTER_LINEAR)
        xywh, cls_ids, confs, xyxy = det.detect(img)
        t1 = time.perf_counter()
        times.append((t_grab - t0, t1 - t_grab))
        print('第 %d 帧: %dx%d, 抓屏 %.0f ms + 推理 %.0f ms, 目标 %d 个'
              % (i + 1, img.shape[1], img.shape[0],
                 (t_grab - t0) * 1000, (t1 - t_grab) * 1000, len(xywh)))
        last_img, last_det = img, (xywh, cls_ids, confs, xyxy)

    xywh, cls_ids, confs, xyxy = last_det
    print('\n最后一帧的目标:')
    counts = {}
    for box_xywh, cid, conf in zip(xywh, cls_ids, confs):
        name = NAMES[cid]
        counts[name] = counts.get(name, 0) + 1
        print('  %-10s conf=%.2f  center=(%.0f,%.0f)  size=(%.0fx%.0f)'
              % (name, conf, box_xywh[0], box_xywh[1], box_xywh[2], box_xywh[3]))
    print('\n各类别数量:', counts if counts else '（什么都没检到）')

    for box, cid, conf in zip(xyxy, cls_ids, confs):
        cv2.rectangle(last_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                      (0, 255, 0), 2)
        cv2.putText(last_img, '%s %.2f' % (NAMES[cid], conf),
                    (int(box[0]), max(int(box[1]) - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    p = os.path.join(out_dir, 'game_frame.png')
    cv2.imwrite(p, last_img)
    print('带框图已存 :', p)

    g = np.mean([t[0] for t in times]) * 1000
    d = np.mean([t[1] for t in times]) * 1000
    print('平均: 抓屏 %.1f ms + 检测 %.1f ms' % (g, d))
    n = len(xywh)
    if n == 0:
        print('\n[注意] 实机画面一个目标都没检到 —— 窗口尺寸/画面可能和训练数据差异太大。')
        return 1
    print('\n[OK] 实机画面检测正常')
    return 0


if __name__ == '__main__':
    sys.exit(main())
