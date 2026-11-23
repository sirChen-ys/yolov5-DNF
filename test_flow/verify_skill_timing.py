# -*- coding: utf-8 -*-
"""实机验证出关节拍：14 秒内应看到 技能(0s) → X(2s) → 技能(4s) → X(6s) ...

会在游戏里真的按键（放技能 + 按 X），但只是验证节拍，不涉及移动。

用法（项目根目录）：
    python test_flow/verify_skill_timing.py
"""
import os
import sys
import time

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import directkeys  # noqa: E402
from game_window import activate_game_window, get_game_region  # noqa: E402
from grabscreen import grab_screen  # noqa: E402
from skill_bar import SkillBar  # noqa: E402

REGION = get_game_region()
DURATION = 14.0
SKILL_INTERVAL = 4.0
X_OFFSET = 2.0


def main():
    if not activate_game_window(countdown=5):
        print('游戏不在前台，退出')
        return 1
    time.sleep(1.0)

    bar = SkillBar(ready_ratio=0.82, min_interval=0.10, repeat_lockout=0.35,
                   skill_interval=SKILL_INTERVAL, basic_x_offset=X_OFFSET)

    # 预热：建立每格的亮度基准
    for _ in range(5):
        bar.ready_map(cv2.cvtColor(grab_screen(REGION), cv2.COLOR_BGRA2BGR))
        time.sleep(0.05)

    print('开始计时 %.0f 秒，节拍 = 每 %.0fs 技能 + 第 %.0fs 补 X' % (DURATION, SKILL_INTERVAL, X_OFFSET))
    t0 = time.time()
    events = []
    while time.time() - t0 < DURATION:
        img0 = grab_screen(REGION)
        key = bar.attack_tick(img0, directkeys.key_press)
        if key:
            events.append((time.time() - t0, key))
        time.sleep(0.02)

    print('\n实际按键时间线:')
    for ts, k in events:
        kind = '普通攻击' if k == 'X' else '技能'
        print('    t=%5.2fs  %-10s %s' % (ts, k, kind))

    # 校验：技能应落在 0/4/8/12 秒附近，X 落在 2/6/10 秒附近
    print('\n校验（容差 ±1.2s）:')
    fails = 0
    for want_t, want_kind in [(0, 'skill'), (2, 'x'), (4, 'skill'), (6, 'x'),
                              (8, 'skill'), (10, 'x'), (12, 'skill')]:
        ok = False
        for ts, k in events:
            is_x = (k == 'X')
            kind_match = (want_kind == 'x') == is_x
            if kind_match and abs(ts - want_t) <= 1.2:
                ok = True
                break
        print('    t≈%2ds 期望 %-5s -> %s' % (want_t, 'X' if want_kind == 'x' else '技能',
                                            '[OK]' if ok else '[FAIL]'))
        if not ok:
            fails += 1

    n_skill = sum(1 for _, k in events if k != 'X')
    n_x = sum(1 for _, k in events if k == 'X')
    print('\n合计: 技能 %d 次, X %d 次（%.0f 秒内）' % (n_skill, n_x, DURATION))
    print('结果:', '节拍正确' if fails == 0 else '%d 个时间点不符' % fails)
    return 0 if fails == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
