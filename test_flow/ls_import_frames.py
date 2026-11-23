# -*- coding: utf-8 -*-
"""把一批抽好的帧导入 Label Studio，并（可选）新建项目。

背景
----
本机 `E:\\Work\\LabelStudio\\api-token.txt` 里那个 token 在 Label Studio 1.23 已失效
（legacy token 被移除，Bearer/Token 都是 401）。所以这里不调 REST API，而是通过
`label-studio shell`（Django ORM）直接建项目 + 任务 —— 和你 export-yolo.ps1 用的是同一套机制。

**关键：光建 Task 是不够的。**
`/data/upload/...` 不是静态目录，而是 `data_import.api.UploadedFileResponse` 视图，
它要 `FileUpload.objects.filter(file=...).last()` 查记录；查不到的话后面
`None.has_permission(...)` 直接 AttributeError(500)，浏览器就报
「There was an issue loading URL from $image value」。
所以每个文件都要建一条 `data_import_fileupload`，并把它的 id 回填到 task.data['upload_id']。

用法
----
    # 新建项目导入
    python test_flow/ls_import_frames.py --frames E:\\Work\\LabelStudio\\data\\media\\frames\\doors01 --title "DNF-YOLO doors01"

    # 追加到已有项目（按文件名去重，可重复执行）
    python test_flow/ls_import_frames.py --frames <目录> --project-id 6
"""
import argparse
import os
import subprocess
import sys

LS_ROOT = r'E:\Work\LabelStudio'
LS_EXE = os.path.join(LS_ROOT, '.venv', 'Scripts', 'label-studio.exe')
LS_BASE = os.path.join(LS_ROOT, 'data')
UPLOAD_ROOT = os.path.join(LS_BASE, 'media', 'upload')

INNER = r'''
import glob, hashlib, os, shutil
from projects.models import Project
from data_import.models import FileUpload
from tasks.models import Task
from users.models import User

FRAME_DIR = {frame_dir!r}
TITLE = {title!r}
TEMPLATE_ID = {template_id}
PROJECT_ID = {project_id}
USER_ID = {user_id}
UPLOAD_ROOT = {upload_root!r}

print('[[LS]] start')
user = User.objects.get(id=USER_ID)

if PROJECT_ID:
    proj = Project.objects.get(id=PROJECT_ID)
    print('[[LS]] append to existing project id=%d' % proj.id)
else:
    src = Project.objects.get(id=TEMPLATE_ID)
    proj = Project.objects.filter(title=TITLE).first()
    if proj is None:
        proj = Project.objects.create(title=TITLE, label_config=src.label_config,
                                      created_by_id=USER_ID, organization_id=src.organization_id,
                                      description='imported by ls_import_frames.py')
        print('[[LS]] created project id=%d' % proj.id)
    else:
        print('[[LS]] reuse project id=%d' % proj.id)

base = os.path.join(UPLOAD_ROOT, str(proj.id))
os.makedirs(base, exist_ok=True)

files = sorted(glob.glob(os.path.join(FRAME_DIR, '*.jpg')) + glob.glob(os.path.join(FRAME_DIR, '*.png')))
print('[[LS]] frames found=%d' % len(files))

existing = set()
for d in Task.objects.filter(project=proj).values_list('data', flat=True):
    if isinstance(d, dict) and d.get('file_name'):
        existing.add(d['file_name'])

created = skipped = 0
for f in files:
    name = os.path.basename(f)
    if name in existing:
        skipped += 1
        continue
    h = hashlib.md5((name + str(proj.id)).encode('utf-8')).hexdigest()[:8]
    dst_name = '%s-%s' % (h, name)
    dst = os.path.join(base, dst_name)
    if not os.path.exists(dst):
        shutil.copy2(f, dst)

    rel = 'upload/%d/%s' % (proj.id, dst_name)
    fu = FileUpload.objects.filter(project=proj, file=rel).first()
    if fu is None:
        fu = FileUpload.objects.create(project=proj, user_id=USER_ID, file=rel)

    Task.objects.create(project=proj,
                        data={{'image': '/data/upload/%d/%s' % (proj.id, dst_name),
                               'file_name': name,
                               'upload_id': fu.id}})
    created += 1

total = Task.objects.filter(project=proj).count()
nfu = FileUpload.objects.filter(project=proj).count()
print('[[LS]] created=%d skipped=%d total=%d fileupload=%d' % (created, skipped, total, nfu))
print('[[LS]] PROJECT_ID=%d' % proj.id)
print('[[LS]] done')
'''


def main():
    ap = argparse.ArgumentParser(description='把帧导入 Label Studio（走 Django ORM，不需要 API token）')
    ap.add_argument('--frames', required=True, help='帧目录')
    ap.add_argument('--title', default=None, help='新项目标题（不填则用目录名）')
    ap.add_argument('--template-project', type=int, default=4, help='复制哪个项目的标注配置')
    ap.add_argument('--project-id', type=int, default=0, help='追加到已有项目（填了就忽略 --title）')
    ap.add_argument('--user-id', type=int, default=2, help='挂在哪个账号下（默认 2 = admin@local.test）')
    args = ap.parse_args()

    if not os.path.isdir(args.frames):
        print('[ERROR] 帧目录不存在:', args.frames)
        return 2
    if not os.path.exists(LS_EXE):
        print('[ERROR] 找不到 label-studio.exe:', LS_EXE)
        return 2

    title = args.title or os.path.basename(os.path.normpath(args.frames))

    inner = INNER.format(frame_dir=args.frames, title=title,
                         template_id=args.template_project,
                         project_id=args.project_id,
                         user_id=args.user_id,
                         upload_root=UPLOAD_ROOT)
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_ls_import_tmp.py')
    with open(script_path, 'w', encoding='utf-8') as fh:
        fh.write(inner)

    # label-studio shell 是交互式 console，逐行解析；多行块会解析错，
    # 所以整段用 exec(compile(...)) 一行喂进去。
    boot = "exec(compile(open(r'%s','rb').read(),'imp.py','exec'))" % script_path

    env = dict(os.environ)
    env['LABEL_STUDIO_BASE_DATA_DIR'] = LS_BASE
    env['PYTHONUTF8'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'

    print('导入中（通过 label-studio shell）…')
    proc = subprocess.run([LS_EXE, 'shell'], input=boot, env=env,
                          capture_output=True, text=True, encoding='utf-8',
                          errors='replace', timeout=900)

    out = (proc.stdout or '') + (proc.stderr or '')
    project_id = None
    for line in out.splitlines():
        if '[[LS]]' in line or 'Traceback' in line or 'Error' in line:
            print(line.strip())
        if 'PROJECT_ID=' in line:
            try:
                project_id = int(line.split('PROJECT_ID=')[1].strip())
            except Exception:
                pass

    try:
        os.remove(script_path)
    except Exception:
        pass

    if project_id:
        print('\n完成。打开: http://localhost:8080/projects/%d' % project_id)
        print('导出:   export-yolo.ps1 -Project %d -SplitByGroup   (会累加进 datasets/dnf)' % project_id)
        return 0
    print('\n[ERROR] 没拿到项目 id，请检查上面的输出')
    return 1


if __name__ == '__main__':
    sys.exit(main())
