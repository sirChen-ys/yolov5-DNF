# -*- coding: utf-8 -*-
"""把小地图区域裁出来放大，确认房间格子/连线/当前位置的渲染方式。"""
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 项目根
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from game_window import activate_game_window, describe_region, get_game_region  # noqa: E402
from grabscreen import grab_screen  # noqa: E402

activate_game_window(countdown=5)
img = cv2.cvtColor(grab_screen(get_game_region()), cv2.COLOR_BGRA2BGR)
print('画面', img.shape)

out = os.path.join(ROOT, 'runs', 'test_flow', 'game')
os.makedirs(out, exist_ok=True)
cv2.imwrite(os.path.join(out, 'minimap_src.png'), img)

# 小地图通常在右上角
x1, y1, x2, y2 = 1150, 5, 1332, 160
crop = img[y1:y2, x1:x2].copy()
big = cv2.resize(crop, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
cv2.imwrite(os.path.join(out, 'minimap_zoom.png'), big)
print('小地图裁剪 (x %d:%d, y %d:%d) -> minimap_zoom.png %s' % (x1, x2, y1, y2, big.shape))
