# -*- coding: utf-8 -*-
"""统计各分类在训练标注里的位置/尺寸分布，用来给误检加几何约束。"""
import collections
import glob
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'datasets', 'dnf')
NAMES = ['hero', 'small_map', 'monster', 'money', 'material', 'door', 'BOSS', 'box', 'options']

W, H = 1332, 750


def main():
    want = sys.argv[1:] or ['options', 'box', 'door']
    stats = {n: {'cx': [], 'cy': [], 'w': [], 'h': [], 'n': 0} for n in want}

    for split in ('train', 'val'):
        for f in glob.glob(os.path.join(DATA, 'labels', split, '*.txt')):
            for line in open(f, encoding='utf-8'):
                p = line.split()
                if len(p) < 5:
                    continue
                c = int(p[0])
                name = NAMES[c]
                if name not in stats:
                    continue
                cx, cy, w, h = (float(v) for v in p[1:5])
                s = stats[name]
                s['cx'].append(cx * W)
                s['cy'].append(cy * H)
                s['w'].append(w * W)
                s['h'].append(h * H)
                s['n'] += 1

    print('画面 %dx%d\n' % (W, H))
    for name in want:
        s = stats[name]
        if not s['n']:
            print('%s: 没有标注' % name)
            continue
        print('---- %s (%d 个) ----' % (name, s['n']))
        for field, label in (('cx', '中心x'), ('cy', '中心y'), ('w', '宽'), ('h', '高')):
            v = s[field]
            print('  %s: min %6.0f  中位 %6.0f  max %6.0f'
                  % (label, min(v), statistics.median(v), max(v)))
        # 归一化到画面比例，方便写约束
        print('  归一化中心: x %.2f~%.2f  y %.2f~%.2f'
              % (min(s['cx']) / W, max(s['cx']) / W,
                 min(s['cy']) / H, max(s['cy']) / H))
        print()


if __name__ == '__main__':
    main()
