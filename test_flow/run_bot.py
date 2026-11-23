# -*- coding: utf-8 -*-
"""启动 main2.py 并把它的 stdout/stderr 自己写进 runs/test_flow/bot.log。

为什么不用 PowerShell 的 `*>` 重定向：
OpenCV 启动时会往 stderr 打一条 WARN，PowerShell 会把它包装成 NativeCommandError，
在某些情况下会连带把整个作业终止（实测机器人跑了 5 分钟后突然 exit 1，日志里没有
任何 Python 异常）。改成 Python 自己接管输出，PowerShell 那边就永远是干净的。

用法（后台跑）：
    python test_flow/run_bot.py
前台调试时想同时看到输出：
    python test_flow/run_bot.py --tee
"""
import os
import runpy
import sys


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
            except Exception:
                pass

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    if root not in sys.path:
        sys.path.insert(0, root)

    log_path = os.path.join(root, 'runs', 'test_flow', 'bot.log')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log = open(log_path, 'w', encoding='utf-8', buffering=1)   # 行缓冲

    console = sys.__stdout__
    if '--tee' in sys.argv:
        sys.stdout = Tee(console, log)
        sys.stderr = Tee(sys.__stderr__, log)
    else:
        sys.stdout = log
        sys.stderr = log

    print('=== bot start ===')
    try:
        runpy.run_path(os.path.join(root, 'main2.py'), run_name='__main__')
    except SystemExit as exc:
        print('=== bot exit (SystemExit %r) ===' % (exc.code,))
        raise
    except BaseException as exc:          # noqa: BLE001
        import traceback
        traceback.print_exc()
        print('=== bot crashed: %s: %s ===' % (type(exc).__name__, exc))
        raise
    finally:
        try:
            log.flush()
        except Exception:
            pass


if __name__ == '__main__':
    main()
