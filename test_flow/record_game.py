# -*- coding: utf-8 -*-
"""录制游戏窗口客户区（用于标注「门 / 出口」等）。

为什么要单独写一个
------------------
1. 必须录**游戏客户区**、且和数据集同分辨率（1332x750）。用整屏或手动框选很容易
   差一点，标出来的框到线上就错位。
2. 屏幕是 125% 缩放时，DPI 不感知的进程会把 1332x750 算成 1066x600，录出来是缩小版。
   这里复用 game_window 里的 DPI 感知逻辑，坐标一律用物理像素。
3. ffmpeg 用 gdigrab 抓的是**桌面可见像素** —— 游戏窗口被别的窗口挡住时，录到的是
   遮挡窗口。所以脚本会先倒计时，让你把游戏切到前台再开始。

用法（项目根目录，双击 test_flow\\record_game.bat 等价于默认参数）：

    python test_flow/record_game.py                    # 录 10 分钟
    python test_flow/record_game.py --seconds 300      # 录 5 分钟
    python test_flow/record_game.py --out E:\\GameData\\raw\\doors02.mp4
    python test_flow/record_game.py --list             # 只看窗口几何，不录

录制结束后会打印文件大小，并抽 3 帧存到 runs/test_flow/rec_preview/ 方便先看一眼。
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from game_window import (activate_game_window, describe_region, find_game_window,
                         get_game_region, is_game_foreground)  # noqa: E402

FFMPEG_CANDIDATES = [
    'ffmpeg',
    r'E:\ffmpeg\ffmpeg-7.1-full_build\bin\ffmpeg.exe',
]
FFPROBE_CANDIDATES = [
    'ffprobe',
    r'E:\ffmpeg\ffmpeg-7.1-full_build\bin\ffprobe.exe',
]


def find_exe(cands):
    for c in cands:
        p = shutil.which(c) if not os.path.isabs(c) else (c if os.path.exists(c) else None)
        if p:
            return p
    return None


def main():
    ap = argparse.ArgumentParser(description='录制游戏窗口客户区')
    ap.add_argument('--out', default=None, help='输出 mp4，默认 E:\\GameData\\raw\\doors_<时间>.mp4')
    ap.add_argument('--seconds', type=int, default=600, help='录制时长（秒），默认 600')
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--crf', type=int, default=20, help='画质，越小越清晰（18~23 合适）')
    ap.add_argument('--countdown', type=int, default=5, help='开始前倒计时秒数')
    ap.add_argument('--list', action='store_true', help='只打印窗口几何，不录制')
    ap.add_argument('--no-activate', action='store_true', help='不要自动切前台')
    args = ap.parse_args()

    ffmpeg = find_exe(FFMPEG_CANDIDATES)
    ffprobe = find_exe(FFPROBE_CANDIDATES)
    if not ffmpeg:
        print('[ERROR] 找不到 ffmpeg')
        return 2

    hwnd = find_game_window()
    if not hwnd:
        print('[ERROR] 没找到游戏窗口（标题「地下城与勇士」/ 类名 Dungeon & Fighter）')
        return 2

    region = get_game_region()
    left, top, x2, y2 = region
    w, h = x2 - left + 1, y2 - top + 1
    print('游戏窗口句柄 : %s' % hex(hwnd))
    print('客户区(物理) : %s' % describe_region(region))
    print('录制尺寸     : %dx%d' % (w, h))
    if (w, h) != (1332, 750):
        print('[注意] 和训练数据 1332x750 不一致，标注后线上可能错位。'
              '建议把游戏窗口调成 1332x750 再录。')
    if args.list:
        print('(--list 模式，不录制)')
        return 0

    if not args.no_activate:
        activate_game_window(countdown=3)
    if not is_game_foreground(hwnd):
        print('[ERROR] 游戏不在前台，录到的会是遮挡它的窗口。请先把游戏点到前台再运行。')
        return 2

    out = args.out
    if not out:
        stamp = time.strftime('%Y%m%d_%H%M%S')
        out = os.path.join('E:\\GameData\\raw', 'doors_%s.mp4' % stamp)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out):
        os.remove(out)

    if w % 2:
        w -= 1
    if h % 2:
        h -= 1

    cmd = [
        ffmpeg, '-hide_banner', '-y', '-loglevel', 'warning',
        '-f', 'gdigrab', '-framerate', str(args.fps),
        '-offset_x', str(left), '-offset_y', str(top),
        '-video_size', '%dx%d' % (w, h),
        '-i', 'desktop',
        '-t', str(args.seconds),
        '-c:v', 'libx264', '-preset', 'veryfast', '-crf', str(args.crf),
        '-pix_fmt', 'yuv420p', out,
    ]

    print('\n=== 即将录制 %d 秒（%.1f 分钟）===' % (args.seconds, args.seconds / 60.0))
    print('输出: %s' % out)
    for i in range(args.countdown, 0, -1):
        print('  %d 秒后开始……请勿切到其它窗口' % i, end='\r', flush=True)
        time.sleep(1)
    print(' ' * 40, end='\r')
    print('开始录制！现在去跑全图的门，跑完等它自动停（不要切窗口）。\n')

    t0 = time.time()
    proc = subprocess.Popen(cmd)
    try:
        proc.wait()
    except KeyboardInterrupt:
        print('\n收到中断，正在让 ffmpeg 收尾（不要强制结束进程，否则文件会损坏）……')
        proc.terminate()
        proc.wait()

    dur = time.time() - t0
    print('\n录制结束，耗时 %.1f 秒，ffmpeg 退出码 %s' % (dur, proc.returncode))
    if not os.path.exists(out):
        print('[ERROR] 没有生成文件，录制失败')
        return 1

    size_mb = os.path.getsize(out) / 1024.0 / 1024.0
    print('文件: %s (%.1f MB)' % (out, size_mb))

    if ffprobe:
        try:
            r = subprocess.run([ffprobe, '-v', 'error', '-select_streams', 'v:0',
                                '-show_entries', 'stream=width,height,nb_frames',
                                '-of', 'csv=p=0', out],
                               capture_output=True, text=True, timeout=30)
            print('分辨率/帧数: %s' % r.stdout.strip())
        except Exception as exc:
            print('ffprobe 失败:', exc)

    # 抽 3 帧存图，方便先确认录到的是游戏而不是遮挡窗口
    prev_dir = os.path.join(ROOT, 'runs', 'test_flow', 'rec_preview')
    os.makedirs(prev_dir, exist_ok=True)
    for i, ts in enumerate([2, args.seconds // 2, max(2, args.seconds - 3)]):
        p = os.path.join(prev_dir, 'frame_%02d.png' % i)
        try:
            subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y',
                            '-ss', str(ts), '-i', out, '-frames:v', '1', p],
                           timeout=60)
        except Exception:
            pass
    print('抽样帧已存到: %s' % prev_dir)
    print('请先看一眼抽样帧，确认是游戏画面（不是浏览器/桌面）再去标注。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
