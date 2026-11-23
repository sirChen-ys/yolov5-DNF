# -*- coding: utf-8 -*-
"""游戏窗口定位 + DPI 感知。

为什么要这个文件
----------------
显示器开了缩放（例如 1920x1080 @125%）时，Python 默认是「DPI 不感知」进程：
  * GetSystemMetrics 报 1536x864 而不是 1920x1080；
  * GetWindowRect 报的坐标是缩小后的（游戏窗口 1332x750 会被报成 1066x600）；
  * grab_screen 抓到的桌面也是缩小 20% 的。
结果就是：模型看到的画面比训练数据（1332x750）小 20%，识别效果明显变差。

所以必须在**任何抓屏/建窗口之前**打开 DPI 感知。grabscreen.py 在 import 时就会调用
enable_dpi_awareness()，所以只要引用了 grab_screen 就自动生效。

用法
----
    from game_window import get_game_region, find_game_window, describe_region

    region = get_game_region(fallback=(0, 0, 1279, 799))
    img0 = grab_screen(region)          # BGRA，尺寸就是游戏客户区
"""

import ctypes

GAME_CLASS = 'Dungeon & Fighter'
GAME_TITLE = '地下城与勇士'

_dpi_aware = False


def enable_dpi_awareness():
    """打开 DPI 感知（Per-Monitor V2 优先）。可重复调用，只生效一次。"""
    global _dpi_aware
    if _dpi_aware:
        return True
    # Per-Monitor Aware V2（Windows 10 1703+）
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        _dpi_aware = True
        return True
    except Exception:
        pass
    # Per-Monitor Aware（Windows 8.1+）
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        _dpi_aware = True
        return True
    except Exception:
        pass
    # System Aware（Vista+）
    try:
        ctypes.windll.user32.SetProcessDPIAware()
        _dpi_aware = True
        return True
    except Exception:
        return False


def is_dpi_aware():
    try:
        return bool(ctypes.windll.user32.GetAwarenessFromDpiAwarenessContext(
            ctypes.windll.user32.GetThreadDpiAwarenessContext()))
    except Exception:
        return _dpi_aware


def find_game_window():
    """返回游戏窗口句柄，找不到返回 None。"""
    enable_dpi_awareness()
    try:
        import win32gui
    except ImportError:
        return None
    try:
        hwnd = win32gui.FindWindow(GAME_CLASS, None)
        if not hwnd:
            hwnd = win32gui.FindWindow(None, GAME_TITLE)
        return hwnd or None
    except Exception:
        return None


def get_game_region(fallback=None):
    """返回抓屏用的游戏客户区 (left, top, x2, y2)。

    注意 grab_screen 内部按 width = x2 - left + 1 算宽度，
    所以这里 x2/y2 用的是「含末像素」的闭区间坐标，抓出来正好是客户区大小。

    找不到游戏窗口时返回 fallback（不传则返回 None）。
    """
    enable_dpi_awareness()
    hwnd = find_game_window()
    if not hwnd:
        return fallback
    try:
        import win32gui
        cw, ch = win32gui.GetClientRect(hwnd)[2:4]
        left, top = win32gui.ClientToScreen(hwnd, (0, 0))
        if cw <= 0 or ch <= 0:
            return fallback
        return (left, top, left + cw - 1, top + ch - 1)
    except Exception:
        return fallback


def describe_region(region):
    if not region:
        return '未找到游戏窗口'
    left, top, x2, y2 = region
    return '屏幕 (%d, %d) - (%d, %d)，尺寸 %dx%d' % (
        left, top, x2 + 1, y2 + 1, x2 - left + 1, y2 - top + 1)


def is_game_foreground(hwnd=None):
    """游戏窗口是否在前台。

    按键（SendInput）是发给当前前台窗口的：游戏不在前台时按下去，
    按键会打到别的窗口上（往聊天框里乱打字、甚至触发别的程序的快捷键）。
    """
    try:
        import win32gui
    except ImportError:
        return True          # 没装 pywin32 就没法判断，不拦
    hwnd = hwnd or find_game_window()
    if not hwnd:
        return False
    try:
        return win32gui.GetForegroundWindow() == hwnd
    except Exception:
        return False


def activate_game_window(countdown=5, verbose=True):
    """把游戏窗口切到前台；切不过去就倒计时让用户自己点。

    :return: True 表示游戏现在确实在前台，可以安全地发按键。
    """
    enable_dpi_awareness()
    hwnd = find_game_window()
    if not hwnd:
        if verbose:
            print('[warn] 没找到游戏窗口，跳过切前台')
        return False

    if _try_force_foreground(hwnd) and verbose:
        print('已把游戏窗口切到前台')

    import time
    try:
        import win32gui
        fg = win32gui.GetForegroundWindow
    except Exception:
        fg = lambda: hwnd  # noqa: E731

    if fg() == hwnd:
        return True

    if verbose:
        print('无法自动切到前台。请在 %d 秒内用鼠标点一下游戏窗口……' % countdown)
    for i in range(countdown, 0, -1):
        if verbose:
            print('  %d' % i, end='\r', flush=True)
        time.sleep(1)
        if fg() == hwnd:
            if verbose:
                print('\n已切到游戏窗口，开始。')
            return True

    if verbose:
        print('\n[ERROR] 游戏窗口仍不在前台。为避免把按键发到别的窗口，已中止。')
        print('        请先点一下游戏窗口再重新运行本脚本。')
    return False


def _try_force_foreground(hwnd):
    """尽力把窗口切到前台。

    Windows 只允许「当前前台进程」抢焦点，所以先 AttachThreadInput 到前台线程，
    不行再补一次模拟 ALT（按下再抬起）解除前台锁。
    :return: True 表示已经成功切到前台
    """
    try:
        import win32api
        import win32con
        import win32gui
        import win32process
    except ImportError:
        return False

    def is_fg():
        try:
            return win32gui.GetForegroundWindow() == hwnd
        except Exception:
            return False

    if is_fg():
        return True

    for attempt in range(2):
        try:
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            win32gui.BringWindowToTop(hwnd)

            fg = win32gui.GetForegroundWindow()
            cur_tid = win32api.GetCurrentThreadId()
            fg_tid = win32process.GetWindowThreadProcessId(fg)[0] if fg else 0

            attached = False
            if fg_tid and fg_tid != cur_tid:
                try:
                    win32process.AttachThreadInput(cur_tid, fg_tid, True)
                    attached = True
                except Exception:
                    attached = False
            try:
                if attempt == 1:
                    # 模拟一次 ALT，解除「前台锁定」限制
                    win32api.keybd_event(win32con.VK_MENU, 0, 0, 0)
                    win32api.keybd_event(win32con.VK_MENU, 0, win32con.KEYEVENTF_KEYUP, 0)
                win32gui.SetForegroundWindow(hwnd)
                win32gui.SetFocus(hwnd)
            except Exception:
                pass
            finally:
                if attached:
                    try:
                        win32process.AttachThreadInput(cur_tid, fg_tid, False)
                    except Exception:
                        pass
        except Exception:
            pass

        if is_fg():
            return True
        try:
            import time
            time.sleep(0.2)
        except Exception:
            pass

    return is_fg()


if __name__ == '__main__':
    import sys

    enable_dpi_awareness()
    print('DPI 感知       :', is_dpi_aware())
    print('屏幕物理分辨率 :', ctypes.windll.user32.GetSystemMetrics(0), 'x',
          ctypes.windll.user32.GetSystemMetrics(1))
    hwnd = find_game_window()
    print('游戏窗口句柄   :', hex(hwnd) if hwnd else '未找到')
    print('游戏客户区     :', describe_region(get_game_region()))

    if '--activate' in sys.argv:
        ok = activate_game_window(countdown=10)
        print('切前台结果     :', '成功' if ok else '失败')
        sys.exit(0 if ok else 1)
