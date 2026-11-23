# -*- coding: utf-8 -*-
"""
DNF 纯检测演示（只画框，不操作游戏）。

替代旧 yolo5_detect.py，使用 ultralytics YOLOv11 推理。
运行后按 P 暂停/继续，按 q 退出。
"""

import random
import time

import cv2

from game_window import describe_region, get_game_region
from getkeys import key_check
from grabscreen import grab_screen
from yolo_detector import YoloDetector, NAMES


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


weights = r'best.pt'
# 抓屏区域：自动对准游戏窗口客户区（不在 (0,0,1280,800) 也能用）。
# 找不到游戏窗口时退回旧值。
window_size = get_game_region(fallback=(0, 0, 1279, 799))
print('抓屏区域:', describe_region(window_size))
conf_thres = 0.3
iou_thres = 0.2
view_img = True
paused = False

names = NAMES
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

detector = YoloDetector(weights, imgsz=640, conf_thres=conf_thres, iou_thres=iou_thres, names=names)

last_time = time.time()
while True:
    if not paused:
        img0 = grab_screen(window_size)
        img = cv2.cvtColor(img0, cv2.COLOR_BGRA2BGR)
        xywh, cls_ids, confs, xyxy = detector.detect(img)
        print('loop took %.3f s, %d objects' % (time.time() - last_time, len(xywh)))
        last_time = time.time()

        for box, cid, conf in zip(xyxy, cls_ids, confs):
            label = '%s %.2f' % (names[cid], conf)
            plot_one_box(box, img, label=label, color=colors[cid], line_thickness=2)

        img = cv2.resize(img, (600, 375))
        if view_img:
            cv2.imshow('window', img)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    keys = key_check()
    if 'P' in keys:
        paused = not paused
        time.sleep(1)
