# -*- coding: utf-8 -*-
"""skill_bar 的离线单元测试（不碰游戏，用合成图）。

    python test_flow/test_skill_bar.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from skill_bar import DEFAULT_ROTATION, SkillBar, slot_rects  # noqa: E402

W, H = 1332, 750


def make_frame(bright_map):
    """按 {key: 亮度} 生成一帧合成画面（灰度值直接写）。"""
    img = np.full((H, W, 3), 8, dtype=np.uint8)
    for k, (x, y, w, h) in slot_rects().items():
        v = bright_map.get(k, 8)
        img[y:y + h, x:x + w] = int(v)
    return img


def all_same(v):
    return {k: v for k in DEFAULT_ROTATION}


class Recorder:
    def __init__(self):
        self.keys = []

    def __call__(self, key):
        self.keys.append(key)


def main():
    fails = 0

    # ---- 用例1：全部就绪，应按轮转顺序 A S D F G H Q W ... 依次放 ----
    bar = SkillBar(rotation=DEFAULT_ROTATION, ready_ratio=0.82,
                   min_interval=0.0, repeat_lockout=0.0, warmup_frames=10)
    ready_img = make_frame(all_same(70))
    for _ in range(12):
        bar.ready_map(ready_img)

    rec = Recorder()
    t = 1000.0
    for i in range(12):
        t += 0.2
        bar.cast(ready_img, rec, now=t)
    print('用例1 轮转顺序:', ' '.join(rec.keys))
    if rec.keys[:6] != DEFAULT_ROTATION[:6]:
        print('  [FAIL] 前 6 个应为 A S D F G H')
        fails += 1
    elif len(set(rec.keys)) != 12:
        print('  [FAIL] 12 次应覆盖 12 个不同技能')
        fails += 1
    else:
        print('  [OK] 12 个技能各放一次，无重复')

    # ---- 用例2：全部变灰（亮度掉到 40%），应回退 X ----
    dark_img = make_frame(all_same(28))     # 70 * 0.4
    rec2 = Recorder()
    for i in range(5):
        t += 0.2
        bar.cast(dark_img, rec2, now=t)
    print('用例2 全变灰时按下的键:', ' '.join(rec2.keys))
    if rec2.keys and set(rec2.keys) == {'X'}:
        print('  [OK] 全部回退到 X（普通攻击）')
    else:
        print('  [FAIL] 应为全部 X')
        fails += 1

    # ---- 用例3：只有部分变灰，应跳过变灰的 ----
    bar2 = SkillBar(rotation=DEFAULT_ROTATION, ready_ratio=0.82,
                    min_interval=0.0, repeat_lockout=0.0, warmup_frames=10)
    for _ in range(12):
        bar2.ready_map(ready_img)
    mixed = all_same(70)
    for k in ['A', 'S', 'D']:          # 前三个冷却
        mixed[k] = 28
    mixed_img = make_frame(mixed)
    rec3 = Recorder()
    t += 0.2
    used = bar2.cast(mixed_img, rec3, now=t)
    print('用例3 A/S/D 变灰时选中的技能:', used)
    if used not in ('A', 'S', 'D') and used != 'X':
        print('  [OK] 跳过了冷却中的技能')
    else:
        print('  [FAIL] 选中了冷却中的技能')
        fails += 1

    # ---- 用例4：技能栏不可见（全黑），不应一直判为冷却 ----
    bar3 = SkillBar(rotation=DEFAULT_ROTATION, ready_ratio=0.82,
                    min_interval=0.0, repeat_lockout=0.0, warmup_frames=10)
    black = np.zeros((H, W, 3), dtype=np.uint8)
    ready_black = bar3.ready_map(black)
    print('用例4 全黑画面判定:', '全部可用' if all(ready_black.values()) else ready_black)
    if all(ready_black.values()):
        print('  [OK] 技能栏不可见时不误判为冷却')
    else:
        print('  [FAIL] 全黑时不应逐格判冷却')
        fails += 1

    # ---- 用例5：出关节拍「每 4 秒一个技能，中间第 2 秒补一次 X」 ----
    bar4 = SkillBar(rotation=DEFAULT_ROTATION, ready_ratio=0.82,
                    min_interval=0.0, repeat_lockout=0.0, warmup_frames=10,
                    skill_interval=4.0, basic_x_offset=2.0)
    for _ in range(12):
        bar4.ready_map(ready_img)

    rec4 = Recorder()
    timeline = [(0.0, 'A'), (0.5, None), (1.5, None), (1.99, None),
                (2.0, 'X'), (2.5, None), (3.5, None), (3.99, None),
                (4.0, 'S'), (5.0, None), (6.0, 'X'), (7.0, None),
                (8.0, 'D'), (10.0, 'X')]
    got = []
    for ts, _want in timeline:
        k = bar4.attack_tick(ready_img, rec4, now=ts)
        got.append(k)

    print('用例5 节拍时间线:')
    ok5 = True
    for (ts, want), g in zip(timeline, got):
        mark = 'OK' if g == want else 'FAIL'
        if g != want:
            ok5 = False
        print('    t=%5.2fs  期望 %-4s 实际 %-4s  %s'
              % (ts, want if want else 'None', g if g else 'None', mark))
    if ok5:
        print('  [OK] 4 秒技能 + 第 2 秒 X 的节拍正确')
    else:
        print('  [FAIL] 节拍不符')
        fails += 1

    # ---- 用例6：节拍起点技能全在冷却 -> 按一次 X，且同一轮不再补 X ----
    bar5 = SkillBar(rotation=DEFAULT_ROTATION, ready_ratio=0.82,
                    min_interval=0.0, repeat_lockout=0.0, warmup_frames=10,
                    skill_interval=4.0, basic_x_offset=2.0)
    for _ in range(12):
        bar5.ready_map(ready_img)
    rec5 = Recorder()
    k0 = bar5.attack_tick(dark_img, rec5, now=20.0)    # 全灰：技能都不可用
    k2 = bar5.attack_tick(dark_img, rec5, now=22.0)    # 第 2 秒
    print('用例6 技能全冷却: t=0 按下 %r, t=2s 按下 %r -> %s'
          % (k0, k2, ' '.join(rec5.keys)))
    if k0 == 'X' and k2 is None:
        print('  [OK] 只按一次 X，不会在同一轮重复补')
    else:
        print('  [FAIL] 冷却时应只在节拍起点按一次 X')
        fails += 1

    print('\n结果:', '全部通过' if fails == 0 else '%d 项失败' % fails)
    return 0 if fails == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
