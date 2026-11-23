# -*- coding: utf-8 -*-
"""Boss / 精英怪血条检测。

为什么需要
----------
打 Boss 时满屏技能特效，YOLO 经常认不出 Boss 也认不出自己（实测 hero 置信度
0.069、monster 0.17），决策层就会「看不到怪」，只会乱走。

但屏幕左上角的 Boss/精英血条是固定 UI：一条几乎横贯屏幕的亮带。
实测在 1332x750 客户区里位于 y≈141~157、x 约 48~712，亮带长度 674 px、亮像素占 94%。
背景场景基本不可能在这个位置出现 450px 以上的连续亮带，所以这个判据很稳。

用法
----
    from boss_bar import has_boss_bar
    if has_boss_bar(img_bgr):
        ...  # 场上有 Boss/精英，直接打
"""

import cv2
import numpy as np

# 血条搜索带（1332x750 客户区坐标）
BAR_X1, BAR_X2 = 0, 760
BAR_Y1, BAR_Y2 = 128, 168

MIN_RUN = 450     # 连续亮像素最短长度
MIN_ROWS = 5      # 连续多少行满足才算有血条
BRIGHT = 60       # 灰度阈值


def _longest_run(row_bool):
    """一维布尔数组里最长连续 True 的长度（向量化，无 Python 循环）。"""
    if not row_bool.any():
        return 0
    a = np.concatenate(([False], row_bool, [False]))
    d = np.diff(a.astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    if len(starts) == 0:
        return 0
    return int((ends - starts).max())


def boss_bar_metrics(img_bgr):
    """返回 (最长亮条, 满足条件的连续行数)，方便调试阈值。"""
    if img_bgr is None:
        return 0, 0
    h, w = img_bgr.shape[:2]
    if h < BAR_Y2 or w < BAR_X2:
        return 0, 0
    band = img_bgr[BAR_Y1:BAR_Y2, BAR_X1:BAR_X2]
    if band.ndim == 3:
        band = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY) if band.shape[2] == 3 \
            else cv2.cvtColor(band, cv2.COLOR_BGRA2GRAY)
    mask = band > BRIGHT
    best_run = 0
    rows = 0
    best_rows = 0
    for r in mask:
        run = _longest_run(r)
        best_run = max(best_run, run)
        if run >= MIN_RUN:
            rows += 1
            best_rows = max(best_rows, rows)
        else:
            rows = 0
    return best_run, best_rows


def has_boss_bar(img_bgr):
    """场上有 Boss/精英（左上血条可见）返回 True。"""
    run, rows = boss_bar_metrics(img_bgr)
    return run >= MIN_RUN and rows >= MIN_ROWS


if __name__ == '__main__':
    import os
    import sys

    ROOT = os.path.dirname(os.path.abspath(__file__))
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)

    args = sys.argv[1:]
    if not args:
        from game_window import get_game_region
        from grabscreen import grab_screen
        img = cv2.cvtColor(grab_screen(get_game_region()), cv2.COLOR_BGRA2BGR)
        run, rows = boss_bar_metrics(img)
        print('当前画面: 最长亮条 %d, 连续行 %d -> 有Boss血条: %s'
              % (run, rows, has_boss_bar(img)))
    else:
        for p in args:
            img = cv2.imread(p)
            if img is None:
                print('%-50s 读不到' % p)
                continue
            run, rows = boss_bar_metrics(img)
            print('%-52s 最长亮条 %4d 连续行 %2d -> %s'
                  % (os.path.basename(p), run, rows, '有Boss' if (run >= MIN_RUN and rows >= MIN_ROWS) else '无'))
