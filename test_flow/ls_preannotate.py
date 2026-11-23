# -*- coding: utf-8 -*-
"""给 Label Studio 项目生成「预标注」(predictions)，方便人工核对而不是从头画。

为什么要这个
------------
第二批 doors01 那 30 张只标了 door，画面里的怪/角色/箱子都没标。
YOLO 会把「没标的目标」当成背景去学 → monster 召回从 0.592 掉到 0.533。
要修就得把这 30 张补全标注；用模型先跑一遍当草稿，人工只做核对/微调。

两段式（关键）
--------------
Label Studio 的 venv 里没有 torch，跑不了模型；我们的项目 venv 里没有 django。
所以拆成两段：
  1) 本项目 venv：读 LS 数据库 → 找到任务对应图片 → 跑 YOLO → 写 JSON
  2) label-studio shell：读 JSON → 写 prediction 表

用法
----
    python test_flow/ls_preannotate.py --project-id 6 --weights best_old.pt --filter doors01
    python test_flow/ls_preannotate.py --project-id 6 --weights best_old.pt --conf 0.35
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import uuid

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

LS_ROOT = r'E:\Work\LabelStudio'
LS_EXE = os.path.join(LS_ROOT, '.venv', 'Scripts', 'label-studio.exe')
LS_BASE = os.path.join(LS_ROOT, 'data')
LS_DB = os.path.join(LS_BASE, 'label_studio.sqlite3')
MEDIA = os.path.join(LS_BASE, 'media')

# 类别 -> 标注控件名（对应项目配置里两个 RectangleLabels）
UNITS = {'hero', 'monster', 'BOSS', 'money', 'material', 'door', 'box'}
UI = {'small_map', 'options'}


def load_tasks(project_id, name_filter=None):
    """从 LS 数据库读任务：返回 [(task_id, file_name, 磁盘路径)]。"""
    import sqlite3
    con = sqlite3.connect('file:%s?mode=ro' % LS_DB, uri=True)
    rows = con.execute('select id, data from task where project_id=? order by id', (project_id,)).fetchall()
    con.close()

    out = []
    for task_id, data in rows:
        if isinstance(data, str):
            data = json.loads(data)
        image = (data or {}).get('image', '')
        fname = (data or {}).get('file_name') or image.split('/')[-1]
        if name_filter and name_filter not in fname:
            continue
        # /data/upload/6/xxx.jpg -> <media>/upload/6/xxx.jpg
        rel = image.replace('/data/', '').lstrip('/')
        path = os.path.join(MEDIA, rel.replace('/', os.sep))
        out.append((task_id, fname, path))
    return out


def build_results(model, img, conf):
    """把一次检测转成 Label Studio rectanglelabels 结果。"""
    from yolo_detector import NAMES
    h, w = img.shape[:2]
    _, cls_ids, confs, xyxy = model.detect(img)
    results = []
    for box, cid, score in zip(xyxy, cls_ids, confs):
        name = NAMES[cid]
        if score < conf:
            continue
        x1, y1, x2, y2 = box
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(w), x2), min(float(h), y2)
        if x2 <= x1 or y2 <= y1:
            continue
        from_name = 'units' if name in UNITS else ('ui' if name in UI else None)
        if from_name is None:
            continue
        results.append({
            'id': str(uuid.uuid4())[:8],
            'from_name': from_name,
            'to_name': 'frame',
            'type': 'rectanglelabels',
            'original_width': w,
            'original_height': h,
            'image_rotation': 0,
            'value': {
                'x': x1 / w * 100.0,
                'y': y1 / h * 100.0,
                'width': (x2 - x1) / w * 100.0,
                'height': (y2 - y1) / h * 100.0,
                'rotation': 0,
                'rectanglelabels': [name],
            },
            'score': float(score),
        })
    return results


INNER = r'''
import json
from tasks.models import Prediction, Task

payload = json.load(open({json_path!r}, encoding='utf-8'))
n_new = n_skip = 0
for item in payload:
    task = Task.objects.get(id=item['task_id'])
    # 先删掉同一模型版本的旧预测，保证可重复执行
    Prediction.objects.filter(task=task, model_version=item['model_version']).delete()
    if not item['results']:
        n_skip += 1
        continue
    Prediction.objects.create(
        task=task,
        project_id=task.project_id,
        result=item['results'],
        score=item.get('score'),
        model_version=item['model_version'],
    )
    n_new += 1
print('[[PRE]] created=%d empty=%d' % (n_new, n_skip))
'''


def main():
    ap = argparse.ArgumentParser(description='给 Label Studio 任务生成预标注')
    ap.add_argument('--project-id', type=int, required=True)
    ap.add_argument('--weights', default='best_old.pt')
    ap.add_argument('--conf', type=float, default=0.30)
    ap.add_argument('--filter', default=None, help='只处理文件名包含该子串的任务')
    args = ap.parse_args()

    from yolo_detector import YoloDetector, NAMES  # noqa: F401

    tasks = load_tasks(args.project_id, args.filter)
    print('待处理任务: %d 个' % len(tasks))
    if not tasks:
        print('[ERROR] 没有匹配的任务')
        return 1

    det = YoloDetector(args.weights, device='cuda:0', imgsz=640,
                       conf_thres=0.05, iou_thres=0.2, names=NAMES)

    payload = []
    total_boxes = 0
    missing = 0
    for task_id, fname, path in tasks:
        img = cv2.imread(path)
        if img is None:
            print('  [warn] 读不到图片:', path)
            missing += 1
            continue
        results = build_results(det, img, args.conf)
        total_boxes += len(results)
        payload.append({
            'task_id': task_id,
            'model_version': 'preannot-%s' % os.path.basename(args.weights),
            'score': None,
            'results': results,
        })

    print('生成预标注: %d 个任务, %d 个框, 缺图 %d' % (len(payload), total_boxes, missing))

    fd, json_path = tempfile.mkstemp(suffix='.json', prefix='ls_preannot_')
    os.close(fd)
    with open(json_path, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, ensure_ascii=False)

    inner_path = json_path + '.py'
    with open(inner_path, 'w', encoding='utf-8') as fh:
        fh.write(INNER.format(json_path=json_path))

    boot = "exec(compile(open(r'%s','rb').read(),'pre.py','exec'))" % inner_path
    env = dict(os.environ)
    env['LABEL_STUDIO_BASE_DATA_DIR'] = LS_BASE
    env['PYTHONUTF8'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'

    print('写入 Label Studio …')
    proc = subprocess.run([LS_EXE, 'shell'], input=boot, env=env,
                          capture_output=True, text=True, encoding='utf-8',
                          errors='replace', timeout=900)
    for line in ((proc.stdout or '') + (proc.stderr or '')).splitlines():
        if '[[PRE]]' in line or 'Traceback' in line or 'Error' in line:
            print(line.strip())

    for p in (json_path, inner_path):
        try:
            os.remove(p)
        except Exception:
            pass

    print('\n完成。打开 http://localhost:8080/projects/%d ，在标注页点「预测」面板'
          '逐条确认/微调即可。' % args.project_id)
    return 0


if __name__ == '__main__':
    sys.exit(main())
