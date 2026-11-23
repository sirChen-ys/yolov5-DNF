# -*- coding: utf-8 -*-
"""离线跑通 main2.py 的「检测 + 决策」流程。

做三件事：
  1. 把 val 集合里的真实游戏截图冒充 grab_screen 的截屏结果；
  2. 屏蔽掉所有真实键鼠输出（SendInput）、窗口显示（imshow）与 sleep；
  3. 原样 exec main2.py 的主循环，跑 N 帧，统计检测结果与决策动作、捕获异常。

全程不会向系统发送任何按键、不会弹窗、不会操作游戏。

用法（在项目根目录执行）：
    python test_flow/simulate_main2.py
    python test_flow/simulate_main2.py --frames 50 --save-every 10
"""

import argparse
import collections
import ctypes
import glob
import os
import sys
import time
import traceback
import types

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 让 main2.py 能 import 同目录下的 directkeys / yolo_detector 等
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class StopLoop(Exception):
    """用来在跑满 N 帧后跳出 main2.py 的 while True。"""


# --------------------------------------------------------------------------
# 伪造 Windows-only 模块
# --------------------------------------------------------------------------
def install_stubs():
    win32api = types.ModuleType('win32api')
    win32api.GetAsyncKeyState = lambda *a, **k: 0
    win32api.GetSystemMetrics = lambda *a, **k: 1920
    win32api.keybd_event = lambda *a, **k: None
    win32api.MapVirtualKey = lambda *a, **k: 0

    win32con = types.ModuleType('win32con')
    win32con.KEYEVENTF_KEYUP = 0x0002
    win32con.SM_CXVIRTUALSCREEN = 78
    win32con.SM_CYVIRTUALSCREEN = 79
    win32con.SM_XVIRTUALSCREEN = 76
    win32con.SM_YVIRTUALSCREEN = 77
    win32con.SRCCOPY = 0x00CC0020

    sys.modules['win32api'] = win32api
    sys.modules['win32con'] = win32con
    sys.modules['win32gui'] = types.ModuleType('win32gui')
    sys.modules['win32ui'] = types.ModuleType('win32ui')


# --------------------------------------------------------------------------
# 伪造 grab_screen：按顺序吐 val 图片，跑满 frames 帧后抛 StopLoop
# --------------------------------------------------------------------------
def install_grabscreen(images, state):
    mod = types.ModuleType('grabscreen')

    def grab_screen(region=None):
        i = state['grabbed']
        if i >= state['frames']:
            raise StopLoop()
        img = images[i % len(images)]
        state['grabbed'] = i + 1
        state['frames_used'].append(os.path.basename(state['image_paths'][i % len(images)]))
        return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    mod.grab_screen = grab_screen
    sys.modules['grabscreen'] = mod
    return mod


# --------------------------------------------------------------------------
# 屏蔽真实键鼠 / 弹窗 / sleep
# --------------------------------------------------------------------------
def silence_windows_io(state):
    """把 SendInput 换成记录器，避免真的往当前窗口发按键。"""
    calls = state['sent_keys']

    def fake_send_input(n_inputs, p_inputs, cb_size):
        try:
            inp = p_inputs.contents
            vk = inp.ii.ki.wVk
            flags = inp.ii.ki.dwFlags
            calls.append(('up' if (flags & 0x0002) else 'down', vk))
        except Exception:
            calls.append(('raw', None))
        return n_inputs

    try:
        ctypes.windll.user32.SendInput = fake_send_input
    except Exception as exc:  # pragma: no cover
        print('[warn] 无法替换 SendInput：', exc)

    real_imshow = cv2.imshow

    def fake_imshow(winname, img):
        state['shown'].append(img.copy())

    cv2.imshow = fake_imshow
    cv2.namedWindow = lambda *a, **k: None
    cv2.destroyAllWindows = lambda *a, **k: None
    cv2.waitKey = lambda *a, **k: -1

    state['real_imshow'] = real_imshow


class Tee:
    """把 main2.py 的 print 收集起来，同时按需转发。"""

    def __init__(self, stream, echo=False, buffer=None):
        self.stream = stream
        self.echo = echo
        self.lines = buffer if buffer is not None else []

    def write(self, s):
        if s:
            self.lines.append(s)
            if self.echo:
                self.stream.write(s)

    def flush(self):
        self.stream.flush()


def main():
    ap = argparse.ArgumentParser(description='离线跑通 main2.py 检测+决策流程')
    ap.add_argument('--frames', type=int, default=50, help='跑多少帧')
    ap.add_argument('--images', default='datasets/dnf/images/val',
                    help='冒充截屏的图片来源目录')
    ap.add_argument('--limit', type=int, default=50, help='最多用多少张图')
    ap.add_argument('--save-every', type=int, default=10, help='每 N 帧存一张带框图')
    ap.add_argument('--echo', action='store_true', help='实时打印 main2 的输出')
    args = ap.parse_args()

    os.chdir(ROOT)
    paths = sorted(glob.glob(os.path.join(args.images, '*.jpg')))[:args.limit]
    if not paths:
        print('没有找到图片：', args.images)
        return 2
    images = [cv2.imread(p) for p in paths]
    print('图片源: %s  (%d 张, 尺寸 %s)' % (args.images, len(images), images[0].shape))

    state = {
        'frames': args.frames,
        'grabbed': 0,
        'frames_used': [],
        'image_paths': paths,
        'sent_keys': [],
        'shown': [],
    }

    install_stubs()
    install_grabscreen(images, state)
    silence_windows_io(state)

    # main2.py 启动时会确认游戏在前台（失败就 SystemExit）；离线模拟不需要，直接放行。
    import game_window
    game_window.activate_game_window = lambda *a, **k: True
    game_window.is_game_foreground = lambda *a, **k: True

    # 记录每帧检测结果
    detect_log = []
    detect_ms = []
    import yolo_detector
    real_detect = yolo_detector.YoloDetector.detect

    def logged_detect(self, img_bgr):
        t = time.perf_counter()
        out = real_detect(self, img_bgr)
        detect_ms.append((time.perf_counter() - t) * 1000.0)
        xywh, cls_ids, confs, xyxy = out
        detect_log.append([self.names[c] for c in cls_ids])
        return out

    yolo_detector.YoloDetector.detect = logged_detect

    real_sleep = time.sleep
    time.sleep = lambda *a, **k: None

    out_dir = os.path.join(ROOT, 'runs', 'test_flow', 'sim')
    os.makedirs(out_dir, exist_ok=True)

    stdout = sys.stdout
    tee = Tee(stdout, echo=args.echo)
    sys.stdout = tee

    error = None
    t0 = time.perf_counter()
    try:
        src_path = os.path.join(ROOT, 'main2.py')
        with open(src_path, encoding='utf-8') as f:
            src = f.read()
        g = {'__name__': '__main__', '__file__': src_path, '__builtins__': __builtins__}
        exec(compile(src, src_path, 'exec'), g)
    except StopLoop:
        pass
    except BaseException as exc:  # noqa: BLE001
        error = exc
        traceback.print_exc(file=tee)
    finally:
        elapsed = time.perf_counter() - t0
        sys.stdout = stdout
        time.sleep = real_sleep
        cv2.imshow = state['real_imshow']

    text = ''.join(tee.lines)
    frames_done = state['grabbed']
    n_detect = len(detect_log)

    # 存几张带框的图
    saved = []
    for idx, frame in enumerate(state['shown']):
        if idx % max(args.save_every, 1) == 0:
            p = os.path.join(out_dir, 'frame_%03d.jpg' % idx)
            cv2.imwrite(p, frame)
            saved.append(p)

    action_keywords = ['释放技能', '普通攻击', '走到门口', '小地图变化', '向右移动', '向左移动', '向上移动', '向下移动',
                       '右上移动', '右下移动', '左上移动', '左下移动',
                       '捡东西', '打开箱子ESC', '进入下一地图', '移动物品到脚下',
                       '再次挑战']
    counts = collections.Counter()
    for kw in action_keywords:
        counts[kw] = text.count(kw)

    class_counter = collections.Counter()
    for classes in detect_log:
        for c in set(classes):
            class_counter[c] += 1

    frames_with_hero = sum(1 for c in detect_log if 'hero' in c)

    print('\n' + '=' * 70)
    print('结果汇总')
    print('=' * 70)
    print('喂帧数            :', frames_done)
    print('实际推理帧数      :', n_detect)
    print('抓屏失败/提前结束 :', '是' if frames_done < args.frames else '否')
    print('总耗时            : %.2f s' % elapsed)
    if n_detect:
        print('平均每帧(含决策)  : %.1f ms' % (elapsed / n_detect * 1000))
    print('检测到 hero 的帧  : %d / %d' % (frames_with_hero, n_detect))
    if detect_ms:
        s = sorted(detect_ms)
        p = lambda q: s[min(len(s) - 1, int(len(s) * q))]
        warm = detect_ms[1:] or detect_ms
        det_sum = sum(detect_ms)
        print('detect() 耗时     : 首帧 %.0f ms | 中位 %.1f ms | p90 %.1f ms | p99 %.1f ms | 最大 %.0f ms'
              % (detect_ms[0], p(0.5), p(0.9), p(0.99), s[-1]))
        print('detect() 均值     : %.1f ms（除首帧 %.1f ms）' % (det_sum / len(detect_ms), sum(warm) / len(warm)))
        print('detect() 总耗时   : %.2f s（占帧循环 %.0f%%）'
              % (det_sum / 1000.0, det_sum / (elapsed * 1000.0) * 100))
        print('最慢 5 帧         :', [round(v) for v in sorted(detect_ms)[-5:]])
        print('理论帧率(仅检测)  : %.1f FPS' % (1000.0 / (det_sum / len(detect_ms))))
    print('\n各类别出现帧数:')
    for name in yolo_detector.NAMES:
        print('  %-10s %d' % (name, class_counter.get(name, 0)))
    print('\n决策动作次数:')
    for kw in action_keywords:
        print('  %-14s %d' % (kw, counts[kw]))
    print('\n模拟发出的按键事件 :', len(state['sent_keys']), '（已拦截，未真正发送）')
    print('存图 :', len(saved), '张 ->', out_dir)

    if error is not None:
        print('\n[FAILED] main2.py 主循环抛出异常: %s: %s' % (type(error).__name__, error))
        return 1
    if frames_done < args.frames:
        print('\n[FAILED] 未跑满 %d 帧（只到 %d 帧）' % (args.frames, frames_done))
        return 1
    if not n_detect:
        print('\n[FAILED] 一帧都没有推理，流程没走起来')
        return 1
    print('\n[OK] main2.py 检测+决策流程跑通 %d 帧，无异常' % frames_done)
    return 0


if __name__ == '__main__':
    sys.exit(main())
