# -*- coding: utf-8 -*-
"""统计数据集每类标签实例数（train/val 分开），用于核对类别平衡。"""
import collections
import glob
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'datasets', 'dnf')
NAMES = ['hero', 'small_map', 'monster', 'money', 'material', 'door', 'BOSS', 'box', 'options']

total_imgs = {}
for split in ('train', 'val'):
    cnt = collections.Counter()
    imgs_with = collections.Counter()
    files = glob.glob(os.path.join(DATA, 'labels', split, '*.txt'))
    for f in files:
        seen = set()
        for line in open(f, encoding='utf-8'):
            line = line.strip()
            if not line:
                continue
            c = int(line.split()[0])
            cnt[NAMES[c]] += 1
            seen.add(NAMES[c])
        for n in seen:
            imgs_with[n] += 1
    total_imgs[split] = len(files)
    if split == 'val':
        val_cnt = cnt
    print('---- %s (%d 张图) ----' % (split, len(files)))
    print('%-10s %10s %10s' % ('class', 'instances', 'images'))
    for n in NAMES:
        print('%-10s %10d %10d' % (n, cnt.get(n, 0), imgs_with.get(n, 0)))
    print('%-10s %10d' % ('TOTAL', sum(cnt.values())))
    print()

print('val 中完全没有实例的类别:',
      [n for n in NAMES if val_cnt.get(n, 0) == 0])
