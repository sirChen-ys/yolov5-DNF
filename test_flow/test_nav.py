# -*- coding: utf-8 -*-
"""nav.py 的离线单测（合成数据，不碰游戏）。

    python test_flow/test_nav.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from nav import (arrived, box_center, choose_door, door_aim_point, hero_foot, merge_doors,
                 walk_toward)  # noqa: E402


def check(name, got, want):
    ok = got == want
    print('%-46s %s  got=%r' % (name, '[OK]  ' if ok else '[FAIL]', got))
    return 0 if ok else 1


def main():
    fails = 0

    # ---- 1) 同一个门口的两根灯柱应合并成一个门 ----
    # 两个 120x230 的框，中心相距 150（灯柱间距），竖直位置相同
    boxes = [[500, 400, 120, 230], [650, 400, 120, 230]]
    merged = merge_doors(boxes)
    fails += check('两根灯柱合并成 1 个门', len(merged), 1)
    if merged:
        c = box_center(merged[0])
        fails += check('合并后中心 x 约为 575', int(round(c[0])), 575)

    # ---- 2) 相隔很远的两个门不应合并 ----
    far = [[300, 400, 120, 230], [1100, 400, 120, 230]]
    fails += check('相隔 800px 的两个门保持独立', len(merge_doors(far)), 2)

    # ---- 3) 三个框（两根灯柱 + 中间门洞）合并成一个 ----
    three = [[500, 400, 120, 230], [650, 400, 120, 230], [590, 420, 60, 200]]
    fails += check('三个重叠框合并成 1 个门', len(merge_doors(three)), 1)

    # ---- 4) 选门：默认不回头，优先右侧 ----
    doors = merge_doors([[300, 400, 120, 230], [1100, 400, 120, 230]])
    hero = (700, 400)
    t = choose_door(doors, hero, img_width=1332, allow_left=False)
    ok = t is not None and t[0] > 700
    print('%-46s %s  target_x=%.0f' % ('不卡住时选右侧的门',
                                       '[OK]  ' if ok else '[FAIL]', t[0] if t else -1))
    fails += 0 if ok else 1

    # ---- 5) 卡住时允许回头，选最近的（这里是左边那扇更近？） ----
    t2 = choose_door(doors, hero, img_width=1332, allow_left=True)
    ok2 = t2 is not None
    print('%-46s %s  target_x=%.0f' % ('卡住时允许左半边门',
                                       '[OK]  ' if ok2 else '[FAIL]', t2[0] if t2 else -1))
    fails += 0 if ok2 else 1

    # ---- 6) 只有左边一扇门时，不卡住就不选 ----
    left_only = merge_doors([[300, 400, 120, 230]])
    t3 = choose_door(left_only, hero, img_width=1332, allow_left=False)
    fails += check('只有左门且未卡住 -> 不选', t3 is None, True)

    # ---- 7) walk_toward 方向映射 ----
    fails += check('目标在右 -> RIGHT', walk_toward((100, 100), (300, 100)), 'RIGHT')
    fails += check('目标在左 -> LEFT', walk_toward((300, 100), (100, 100)), 'LEFT')
    fails += check('目标在下 -> DOWN', walk_toward((100, 100), (100, 300)), 'DOWN')
    fails += check('目标在上 -> UP', walk_toward((100, 300), (100, 100)), 'UP')
    fails += check('目标右下 -> RIGHT_DOWN', walk_toward((100, 100), (300, 300)), 'RIGHT_DOWN')
    fails += check('目标左上 -> LEFT_UP', walk_toward((300, 300), (100, 100)), 'LEFT_UP')
    fails += check('已在死区内 -> None', walk_toward((100, 100), (110, 110)), None)

    # ---- 8) 到达判定：角色进入门框(含 margin) 才算到 ----
    door = (500, 400, 700, 600)
    fails += check('角色在门框内', arrived((600, 500), door), True)
    fails += check('角色在门框外远处', arrived((100, 100), door), False)
    fails += check('角色在门框边缘 margin 内', arrived((480, 500), door, margin=40), True)

    # ---- 9) hero_foot：脚下位置比框中心低 ----
    foot = hero_foot([100, 100, 80, 200])
    fails += check('hero_foot y = 中心 + h*0.5', int(foot[1]), 200)

    # ---- 10) door_aim_point：左右侧门应该瞄「近边」而不是中心 ----
    door = (1000, 300, 1300, 500)      # 右侧一扇门（中心 1150,400）

    # 角色在门左边很远 → 瞄点应靠近门的左边缘(1000)，而不是中心(1150)
    ax, ay = door_aim_point((400, 400), door)
    ok = ax < 1050 and 300 <= ay <= 500
    print('%-46s %s  瞄点=(%.0f,%.0f) 中心x=1150' %
          ('角色在门左侧 → 瞄左边缘', '[OK]  ' if ok else '[FAIL]', ax, ay))
    fails += 0 if ok else 1

    # 角色在门右边 → 瞄右边缘
    bx2, by2 = door_aim_point((1500, 400), door)
    ok2 = bx2 > 1250 and 300 <= by2 <= 500
    print('%-46s %s  瞄点=(%.0f,%.0f)' %
          ('角色在门右侧 → 瞄右边缘', '[OK]  ' if ok2 else '[FAIL]', bx2, by2))
    fails += 0 if ok2 else 1

    # 角色在门内 → 瞄点在门内，且朝中心靠
    cx2, cy2 = door_aim_point((1050, 400), door)
    inside = door[0] <= cx2 <= door[2] and door[1] <= cy2 <= door[3]
    closer = abs(cx2 - 1150) < abs(1050 - 1150)
    ok3 = inside and closer
    print('%-46s %s  瞄点=(%.0f,%.0f)' %
          ('角色在门内 → 瞄点朝中心收', '[OK]  ' if ok3 else '[FAIL]', cx2, cy2))
    fails += 0 if ok3 else 1

    # 瞄点永远落在门框内（随机采样验证）
    import random as _rnd
    _rnd.seed(0)
    bad = 0
    for _ in range(200):
        hx, hy = _rnd.uniform(0, 1332), _rnd.uniform(0, 750)
        px, py = door_aim_point((hx, hy), door)
        if not (door[0] - 1 <= px <= door[2] + 1 and door[1] - 1 <= py <= door[3] + 1):
            bad += 1
    fails += check('200 次随机采样：瞄点都在门框内', bad, 0)

    print('\n结果:', '全部通过' if fails == 0 else '%d 项失败' % fails)
    return 0 if fails == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
