# -*- coding: utf-8 -*-
"""实机验证技能轮转 + 冷却判断 + 回退 X。

会在游戏里真的按技能键（城镇里放技能无害）。

用法（项目根目录）：
    python test_flow/verify_skills.py
"""
import os
import sys
import time

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 项目根
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import directkeys  # noqa: E402
from game_window import activate_game_window, get_game_region  # noqa: E402
from grabscreen import grab_screen  # noqa: E402
from skill_bar import SkillBar  # noqa: E402

REGION = get_game_region()


def main():
    if not activate_game_window(countdown=5):
        print('游戏不在前台，退出')
        return 1
    time.sleep(1.0)

    bar = SkillBar(rotation=['A', 'S', 'D', 'F', 'G', 'H', 'Q', 'W', 'E', 'R', 'T', 'Y'],
                   ready_ratio=0.82, min_interval=0.10, repeat_lockout=0.35)

    for _ in range(5):
        bar.ready_map(cv2.cvtColor(grab_screen(REGION), cv2.COLOR_BGRA2BGR))
        time.sleep(0.05)

    seq = []
    for _ in range(45):
        img0 = grab_screen(REGION)
        used = bar.cast(img0, directkeys.key_press)
        if used:
            seq.append(used)
        time.sleep(0.08)

    print('按下序列（45 次尝试）:')
    print('  ' + ' '.join(seq))

    x_count = seq.count('X')
    non_x = [k for k in seq if k != 'X']
    print('\n技能次数: %d   回退 X 次数: %d   用到的不同技能: %s'
          % (len(non_x), x_count, sorted(set(non_x))))

    img = cv2.cvtColor(grab_screen(REGION), cv2.COLOR_BGRA2BGR)
    ready = bar.ready_map(img)
    vals = bar.last_debug.get('values', {})
    ratios = bar.last_debug.get('ratios', {})
    print('\n当前状态:')
    print('%-4s %8s %8s %6s' % ('key', 'bright', 'ratio', 'ready'))
    for k in bar.rotation:
        if k in vals:
            print('%-4s %8.1f %8.2f %6s' % (k, vals[k], ratios.get(k, 1.0), ready.get(k)))

    out = os.path.join(ROOT, 'runs', 'test_flow', 'game', 'skillbar_live.png')
    vis = img.copy()
    for k, (x, y, w, h) in bar.slots.items():
        ok = ready.get(k, True)
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0) if ok else (0, 0, 255), 1)
        cv2.putText(vis, k, (x + 1, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 0) if ok else (0, 0, 255), 1)
    cv2.imwrite(out, cv2.resize(vis[640:750, 500:880].copy(), None, fx=4, fy=4,
                                interpolation=cv2.INTER_NEAREST))
    print('\n存图:', out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
