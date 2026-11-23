# -*- coding: utf-8 -*-
"""技能栏冷却检测 + 技能轮转释放。

为什么需要这个
--------------
原来 main2.py 打怪只会 `key_press("A")`，其它 11 个技能完全不用；
而且技能在冷却（图标变灰）时还在按，等于空放。

做法
----
1. 按固定像素读技能栏 12 个格子（上排 Q W E R T Y，下排 A S D F G H）；
2. 每个格子算亮度，用「该格子历史最亮值」当基准：图标变灰 = 变暗，亮度掉到基准
   的 `ready_ratio` 以下就判定「冷却中」；
3. 打怪时在**可用**的技能里按 `rotation` 轮流转，一个都不可用就按 X（普通攻击）。

坐标是对 1332x750 客户区量出来的（见 game_window.get_game_region），
换分辨率要重新量，可以跑 `python skill_bar.py` 存图核对。
"""

import cv2

# ---- 技能栏几何（1332x750 客户区，实测校准）----
SLOT_X0 = 537          # 第一个技能格左边缘
SLOT_PITCH = 40        # 每格水平间距
SLOT_W = 38
SLOT_H = 36
SLOT_Y_TOP = 662       # 上排 y
SLOT_Y_BOTTOM = 700    # 下排 y

KEYS_TOP = ['Q', 'W', 'E', 'R', 'T', 'Y']
KEYS_BOTTOM = ['A', 'S', 'D', 'F', 'G', 'H']

# 默认轮转顺序：先小技能，后大招
DEFAULT_ROTATION = ['A', 'S', 'D', 'F', 'G', 'H', 'Q', 'W', 'E', 'R', 'T', 'Y']

# 技能栏整体太暗时认为「技能栏不可见」（例如开了别的界面/读图）
BAR_MIN_BRIGHTNESS = 12.0


def slot_rects():
    out = {}
    for i, k in enumerate(KEYS_TOP):
        out[k] = (SLOT_X0 + i * SLOT_PITCH, SLOT_Y_TOP, SLOT_W, SLOT_H)
    for i, k in enumerate(KEYS_BOTTOM):
        out[k] = (SLOT_X0 + i * SLOT_PITCH, SLOT_Y_BOTTOM, SLOT_W, SLOT_H)
    return out


class SkillBar:
    """读技能栏，判断哪些技能可用，并轮流释放。"""

    def __init__(self, rotation=None, ready_ratio=0.82, baseline_decay=0.0005,
                 verbose=False, min_interval=0.0, repeat_lockout=0.35,
                 warmup_frames=45, max_jump_ratio=1.35,
                 skill_interval=4.0, basic_x_offset=2.0):
        """
        :param rotation:      轮转顺序
        :param ready_ratio:   亮度 / 基准亮度 低于这个值即认为冷却中
        :param baseline_decay: 基准每帧缓慢衰减，适应光照变化
        :param min_interval:  两个技能之间的最小间隔（秒）
        :param repeat_lockout: 同一个技能在这段时间内不重复按（秒）。
                              图标变灰有一两帧延迟，靠这个避免连按同一个技能。
        :param warmup_frames: 预热采样帧数，之后用这些样本的 90 分位当基准
                              （直接取最大值会被技能特效的亮闪污染）
        :param max_jump_ratio: 基准向上跟随的最大倍数，超过就当成特效闪光丢掉
        :param skill_interval: 技能节拍（秒）。每这么久放一个技能。
        :param basic_x_offset: 一个节拍里，在第几秒补一次 X（普通攻击）。
        """
        self.slots = slot_rects()
        self.rotation = list(rotation or DEFAULT_ROTATION)
        # 只保留真实存在的槽位
        self.rotation = [k for k in self.rotation if k in self.slots]
        self.ready_ratio = ready_ratio
        self.baseline_decay = baseline_decay
        self.verbose = verbose
        self.min_interval = min_interval
        self.repeat_lockout = repeat_lockout
        self.warmup_frames = warmup_frames
        self.max_jump_ratio = max_jump_ratio
        self.skill_interval = skill_interval
        self.basic_x_offset = basic_x_offset

        self.baseline = {}
        self._warm = {}
        self._frames = 0
        self.idx = 0
        self.last_cast_time = -1e9
        self.last_press = {}
        self.last_debug = {}

        # 节拍状态
        self.cycle_start = None
        self.skill_done = False
        self.x_done = False
        self.last_action = None

    # ---------- 读 ----------
    def measure(self, img_bgr):
        """返回 {key: 亮度}。img_bgr 可以是 BGR 或 BGRA。"""
        if img_bgr is None:
            return {}
        h, w = img_bgr.shape[:2]
        if img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        out = {}
        for k, (x, y, sw, sh) in self.slots.items():
            if x + sw > w or y + sh > h:
                continue
            out[k] = float(gray[y:y + sh, x:x + sw].mean())
        return out

    # ---------- 更新基准 ----------
    @staticmethod
    def _p90(values):
        s = sorted(values)
        if not s:
            return 0.0
        return s[min(len(s) - 1, int(len(s) * 0.9))]

    def update_baseline(self, values):
        """维护每个技能格的「就绪亮度」基准。

        预热期先收集样本（用当前最大值临时当基准，保证预热期间也能判冷却），
        预热结束后改用样本的 90 分位，避免单帧亮闪把基准永久抬高。
        之后只允许小幅向上跟随（max_jump_ratio 内的变化），并极慢向下衰减。
        """
        if self._frames < self.warmup_frames:
            for k, v in values.items():
                self._warm.setdefault(k, []).append(v)
            self._frames += 1
            if self._frames < self.warmup_frames:
                for k, arr in self._warm.items():
                    self.baseline[k] = max(arr)
            else:
                for k, arr in self._warm.items():
                    self.baseline[k] = self._p90(arr)
            return

        for k, v in values.items():
            prev = self.baseline.get(k)
            if prev is None:
                self.baseline[k] = v
            elif v > prev:
                # 正常变亮才跟随；超过 max_jump_ratio 倍视为特效闪光，丢弃
                if v <= prev * self.max_jump_ratio:
                    self.baseline[k] = v
            else:
                self.baseline[k] = prev - prev * self.baseline_decay

    # ---------- 判断可用 ----------
    def ready_map(self, img_bgr):
        """返回 {key: bool}；同时更新基准。技能栏不可见时全部返回 True（避免卡在按 X）。"""
        values = self.measure(img_bgr)
        if not values:
            return {}
        if max(values.values()) < BAR_MIN_BRIGHTNESS:
            self.last_debug = {'visible': False}
            return {k: True for k in values}

        self.update_baseline(values)

        ready = {}
        for k, v in values.items():
            base = self.baseline.get(k, v)
            ratio = (v / base) if base > 1e-6 else 1.0
            ready[k] = ratio >= self.ready_ratio
        self.last_debug = {
            'visible': True,
            'values': values,
            'ratios': {k: (values[k] / self.baseline.get(k, values[k]))
                       for k in values},
        }
        return ready

    # ---------- 选一个技能 ----------
    def pick(self, ready, now):
        """按轮转顺序挑一个「可用且最近没按过」的技能；都不行返回 None。

        同一技能被锁在 repeat_lockout 内不重复按：图标变灰有一两帧延迟，
        不做这个会连按同一个技能四次（实测 Y Y Y Y、A A A A）。
        """
        n = len(self.rotation)
        for i in range(n):
            k = self.rotation[(self.idx + i) % n]
            if not ready.get(k, True):
                continue
            if now - self.last_press.get(k, -1e9) < self.repeat_lockout:
                continue
            self.idx = (self.idx + i + 1) % n
            return k
        return None

    # ---------- 打一次 ----------
    def cast(self, img_bgr, key_press, now=None, fallback='X'):
        """放一个技能；都不行就按 fallback（普通攻击）。返回实际按下的键。"""
        import time
        now = time.time() if now is None else now
        if now - self.last_cast_time < self.min_interval:
            return None

        ready = self.ready_map(img_bgr)
        key = self.pick(ready, now)
        if key is None:
            key = fallback
        key_press(key)
        self.last_cast_time = now
        self.last_press[key] = now
        return key

    # ---------- 按节拍打（推荐给 main2 用） ----------
    def attack_tick(self, img_bgr, key_press, now=None, fallback='X'):
        """按「每 skill_interval 秒放一个技能 + 中间第 basic_x_offset 秒补一次 X」节拍出招。

        节拍（默认 4 秒一轮）：
            t=0s  -> 放一个可用技能（都冷却中就按 X）
            t=2s  -> 按一次 X（普通攻击）
            t=4s  -> 下一轮

        用「周期起点」而不是「距上次技能多久」来判断，冷却里按不出技能时
        不会变成每帧狂按 X。

        :return: 本次按下的键；这一帧不该出招时返回 None
        """
        import time
        now = time.time() if now is None else now

        if self.cycle_start is None or now - self.cycle_start >= self.skill_interval:
            self.cycle_start = now
            self.skill_done = False
            self.x_done = False

        off = now - self.cycle_start

        # 1) 节拍起点：放技能
        if not self.skill_done:
            self.skill_done = True
            ready = self.ready_map(img_bgr)
            key = self.pick(ready, now)
            if key is None:
                key = fallback
                self.x_done = True        # 技能都在冷却，这一轮就不再补 X 了
            key_press(key)
            self.last_cast_time = now
            self.last_press[key] = now
            self.last_action = ('skill', key, off)
            return key

        # 2) 节拍中间：补一次 X
        if not self.x_done and off >= self.basic_x_offset:
            self.x_done = True
            key_press(fallback)
            self.last_press[fallback] = now
            self.last_action = ('x', fallback, off)
            return fallback

        return None


if __name__ == '__main__':
    # 自检：抓一帧，打印 12 个格子的亮度和可用性，并存一张带格子的图
    import os
    import sys

    import numpy as np

    ROOT = os.path.dirname(os.path.abspath(__file__))   # 本文件就在项目根
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)

    from game_window import describe_region, get_game_region
    from grabscreen import grab_screen

    region = get_game_region()
    print('抓屏区域:', describe_region(region))
    img = cv2.cvtColor(grab_screen(region), cv2.COLOR_BGRA2BGR)

    bar = SkillBar(verbose=True)
    ready = bar.ready_map(img)
    vals = bar.last_debug.get('values', {})
    ratios = bar.last_debug.get('ratios', {})
    print('%-4s %8s %8s %6s' % ('key', 'bright', 'ratio', 'ready'))
    for k in DEFAULT_ROTATION:
        if k in vals:
            print('%-4s %8.1f %8.2f %6s' % (k, vals[k], ratios.get(k, 1.0), ready.get(k)))

    vis = img.copy()
    for k, (x, y, w, h) in bar.slots.items():
        ok = ready.get(k, True)
        color = (0, 255, 0) if ok else (0, 0, 255)
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 1)
        cv2.putText(vis, k, (x + 1, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    out = os.path.join(ROOT, 'runs', 'test_flow', 'game')
    os.makedirs(out, exist_ok=True)
    crop = vis[630:760, 480:900].copy()      # copy(): 切片不连续时 imwrite 会静默失败
    big = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    p = os.path.join(out, 'skillbar_ready.png')
    ok = cv2.imwrite(p, big)
    print('存图: %s  (%s)  %s' % (p, big.shape, 'OK' if ok else 'FAILED'))
