# -*- coding: utf-8 -*-
"""
YOLOv11 推理封装（基于 ultralytics）。

替代原项目内置的 vendored YOLOv5 推理（models.experimental / utils.general）。
对外提供一个与旧代码决策逻辑兼容的接口，方便 main2.py 等脚本无缝切换。

用法：
    from yolo_detector import YoloDetector

    det = YoloDetector('best.pt', device='cuda:0', imgsz=640)
    xywh, cls_ids, confs, xyxy = det.detect(img_bgr)   # img_bgr: (H, W, 3) numpy uint8
"""

from ultralytics import YOLO

# 9 类（沿用原项目，与 data/dnf.yaml 保持一致）
NAMES = ['hero', 'small_map', 'monster', 'money', 'material', 'door', 'BOSS', 'box', 'options']


class YoloDetector:
    """封装 ultralytics YOLO 模型，输出与旧决策逻辑兼容的检测结果。"""

    def __init__(self, weights, device=None, imgsz=640, conf_thres=0.3, iou_thres=0.2,
                 agnostic_nms=False, names=None, half=None):
        """
        :param weights:       权重路径（yolo11n.pt / best.pt 等）
        :param device:        None=自动, 或 'cpu' / 'cuda:0' / 0
        :param imgsz:         推理输入尺寸
        :param conf_thres:    置信度阈值
        :param iou_thres:     NMS IoU 阈值
        :param agnostic_nms:  是否类别无关 NMS
        :param names:         类别名列表，默认使用 NAMES
        :param half:          是否 FP16（None=自动）
        """
        self.weights = weights
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.agnostic_nms = agnostic_nms
        self.names = names or NAMES

        self.model = YOLO(weights)
        if device is not None:
            self.model.to(device)
        # 强制 FP16/FP32（可选；ultralytics 在 GPU 上默认 FP16）
        if half is not None:
            self.model.model.half() if half else self.model.model.float()

    def detect(self, img_bgr):
        """
        对单张 BGR 图做推理。

        :param img_bgr: numpy uint8 数组，shape (H, W, 3)，BGR 顺序
        :return: (xywh, cls_ids, confs, xyxy)
                 xywh    -> list[[cx, cy, w, h], ...]（像素坐标，中心点格式）
                 cls_ids -> list[int]（类别 id，对应 self.names）
                 confs   -> list[float]（置信度）
                 xyxy    -> list[[x1, y1, x2, y2], ...]（左上/右下角点）
                 三个列表按 conf 降序排列，长度相等；无目标时返回 ([], [], [], [])
        """
        results = self.model.predict(
            source=img_bgr,
            conf=self.conf_thres,
            iou=self.iou_thres,
            imgsz=self.imgsz,
            agnostic_nms=self.agnostic_nms,
            verbose=False,
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return [], [], [], []

        boxes = results[0].boxes
        xywh = boxes.xywh.cpu().numpy().tolist()
        xyxy = boxes.xyxy.cpu().numpy().tolist()
        cls_ids = boxes.cls.cpu().numpy().astype(int).tolist()
        confs = boxes.conf.cpu().numpy().tolist()
        return xywh, cls_ids, confs, xyxy


if __name__ == '__main__':
    import cv2
    import sys

    if len(sys.argv) < 2:
        print('用法: python yolo_detector.py <weights> [image_path]')
        sys.exit(1)

    weights = sys.argv[1]
    img_path = sys.argv[2] if len(sys.argv) > 2 else None
    det = YoloDetector(weights)

    if img_path:
        img = cv2.imread(img_path)
        xywh, cls_ids, confs, xyxy = det.detect(img)
    else:
        from grabscreen import grab_screen
        img = cv2.cvtColor(grab_screen((0, 0, 1280, 800)), cv2.COLOR_BGRA2BGR)
        xywh, cls_ids, confs, xyxy = det.detect(img)

    for xywh, cid, conf in zip(xywh, cls_ids, confs):
        print(f'{det.names[cid]:<10} conf={conf:.2f}  center=({xywh[0]:.0f},{xywh[1]:.0f}) '
              f'size=({xywh[2]:.0f}x{xywh[3]:.0f})')
