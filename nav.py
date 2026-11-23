# -*- coding: utf-8 -*-
"""房间推进用的几何工具：合并门框、选目标门、朝目标走。

为什么要单独抽出来
------------------
原来 main2.py 里进门逻辑是「四大象限 if/elif 嵌套 + 30px 判定」，有两个硬伤：
  1. 模型对同一个门会出 2~3 个框（两根灯柱各一个），代码取的是**列表最后一个**
     （也就是置信度最低的那个），方向就飘了；
  2. 「进入下一地图」要求角色中心离门中心 < 30px，实际很难正好穿过那个点，
     结果就是永远进不了门（实测三轮一次都没触发）。

这里改成：
  * 先把重叠/相邻的门框合并成一个「门」；
  * 再按「优先右边、其次最近」选一个目标门；
  * 然后用一个通用的 walk_toward() 朝门中心走，进入门框范围就算到达。

另外提供 minimap_signature() / minimap_changed()，用小地图画面变化判断
「是不是换房间了」，用来重置卡死计时。
"""

import numpy as np

# 方向名要跟 directkeys / direction_move 一致
DIRS = ('UP', 'DOWN', 'LEFT', 'RIGHT')


def _box_xyxy(b):
    """接受 [cx,cy,w,h] 或 [x1,y1,x2,y2]，统一转成 (x1,y1,x2,y2)。"""
    if len(b) == 4:
        # 我们的检测输出是中心点格式
        cx, cy, w, h = b
        return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)
    raise ValueError('box 需要 4 个值')


def _overlap_or_close(a, b, gap):
    """两个矩形是否重叠，或水平/垂直间隙小于 gap 且另一维有重叠。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    # 两维都重叠 → 直接合并
    if ax1 <= bx2 and bx1 <= ax2 and ay1 <= by2 and by1 <= ay2:
        return True

    dx = max(ax1 - bx2, bx1 - ax2)      # <0 表示水平重叠
    dy = max(ay1 - by2, by1 - ay2)
    v_overlap = min(ay2, by2) - max(ay1, by1)
    h_overlap = min(ax2, bx2) - max(ax1, bx1)

    # 并排的两个灯柱：水平挨着、竖直方向大量重叠
    if dx <= gap and v_overlap > 0.4 * min(ay2 - ay1, by2 - by1):
        return True
    # 上下叠着的两个框
    if dy <= gap and h_overlap > 0.4 * min(ax2 - ax1, bx2 - bx1):
        return True
    return False


def merge_doors(boxes, gap=70):
    """把属于同一个门的多個框合并成一个。boxes: [[cx,cy,w,h], ...]

    返回合并后的列表，每项 (x1, y1, x2, y2)。
    """
    rects = [_box_xyxy(b) for b in boxes]
    if not rects:
        return []

    changed = True
    while changed:
        changed = False
        out = []
        used = [False] * len(rects)
        for i in range(len(rects)):
            if used[i]:
                continue
            cur = list(rects[i])
            used[i] = True
            for j in range(i + 1, len(rects)):
                if used[j]:
                    continue
                if _overlap_or_close(tuple(cur), rects[j], gap):
                    cur = [min(cur[0], rects[j][0]), min(cur[1], rects[j][1]),
                           max(cur[2], rects[j][2]), max(cur[3], rects[j][3])]
                    used[j] = True
                    changed = True
            out.append(tuple(cur))
        rects = out
    return rects


def box_center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def choose_door(doors, hero_center, img_width, allow_left=False,
                side_pref=0.52, same_level_bonus=1.5):
    """选一个目标门。

    :param doors:       merge_doors 的输出
    :param hero_center: (x, y) 角色脚下位置
    :param allow_left:  False 时不考虑左半边的门（防回头）；卡住时传 True
    :return: (center_x, center_y, box) 或 None
    """
    if not doors:
        return None
    hx, hy = hero_center
    split = img_width * side_pref

    cands = []
    for d in doors:
        cx, cy = box_center(d)
        if not allow_left and cx < split:
            continue
        dist = ((cx - hx) ** 2 + (cy - hy) ** 2) ** 0.5
        # 高度接近的门优先（同一水平线，走过去不会撞墙）
        if abs(cy - hy) < 60:
            dist /= same_level_bonus
        cands.append((dist, cx, cy, d))

    if not cands:
        return None
    cands.sort(key=lambda t: t[0])
    _, cx, cy, d = cands[0]
    return (cx, cy, d)


def hero_foot(hero_xywh, foot_ratio=0.5):
    """英雄的框一般只到上半身，用「脚下」位置去比门/地面物品更准。"""
    cx, cy, w, h = hero_xywh
    return (cx, cy + h * foot_ratio)


def walk_toward(hero_pt, target_pt, dead_zone=35):
    """返回朝目标点走的方向名；已经在死区内返回 None。"""
    dx = target_pt[0] - hero_pt[0]
    dy = target_pt[1] - hero_pt[1]
    if abs(dx) <= dead_zone and abs(dy) <= dead_zone:
        return None
    horiz = None if abs(dx) <= dead_zone else ('RIGHT' if dx > 0 else 'LEFT')
    vert = None if abs(dy) <= dead_zone else ('DOWN' if dy > 0 else 'UP')
    if horiz and vert:
        return '%s_%s' % (horiz, vert)
    return horiz or vert


def arrived(hero_pt, door_box, margin=40):
    """角色是否已经走到门框里（比「离门中心 30px」宽松且更合理）。"""
    x1, y1, x2, y2 = door_box
    hx, hy = hero_pt
    return (x1 - margin) <= hx <= (x2 + margin) and (y1 - margin) <= hy <= (y2 + margin)


def door_key(door_box, cell=80):
    """把门框压成一个粗粒度的 key，用来做「这扇门走不过去」的黑名单。

    用 80px 一格而不是精确坐标：模型每帧的框会轻微抖动，精确 key 会导致
    每次都被当成「新门」，黑名单永远命不中。
    """
    x1, y1, x2, y2 = door_box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return (int(cx) // cell, int(cy) // cell)


def door_aim_point(hero_pt, door_box, inset=0.15):
    """返回朝门口走应该瞄的点：门框上离角色最近的位置，再往框内缩一点。

    为什么不用几何中心：左右两侧的门框常被墙/柱子切掉一半，中心点可能落在墙里
    或者屏幕外。照中心走就会在门口一直顶着墙跑（实测左右门卡死）。
    取「最近边上的点」再往里缩 inset，能保证目标是站得住、进得去的地方。

    :param hero_pt: (x, y) 角色脚下位置
    :param door_box: (x1, y1, x2, y2)
    :param inset: 往框内缩的比例（0.15 = 缩 15%）
    """
    x1, y1, x2, y2 = door_box
    hx, hy = hero_pt
    # 把角色位置夹到矩形范围内 = 矩形上离角色最近的点
    cx = min(max(hx, x1), x2)
    cy = min(max(hy, y1), y2)
    # 再朝中心推 inset，避免只贴着边沿
    bx, by = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return (cx + (bx - cx) * inset, cy + (by - cy) * inset)


# ---------------- 小地图：判断是否换房间 ----------------

def minimap_signature(img_bgr, box, size=(24, 24)):
    """取小地图区域的低分辨率灰度指纹，用来比较「画面有没有换房间」。"""
    if img_bgr is None or box is None:
        return None
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = max(0, min(x1, w - 1))
    x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    gray = crop if crop.ndim == 2 else (crop[:, :, 0] * 0.114 +
                                        crop[:, :, 1] * 0.587 +
                                        crop[:, :, 2] * 0.299)
    return np.array(__import__('cv2').resize(gray.astype(np.float32), size))


def minimap_changed(prev_sig, cur_sig, thresh=12.0):
    """两个指纹的平均绝对差超过阈值 → 认为换了房间。"""
    if prev_sig is None or cur_sig is None:
        return False
    return float(np.abs(prev_sig - cur_sig).mean()) > thresh
