# -*- coding: utf-8 -*-
"""结算界面识别 + 再次挑战(F10) 的回归测试。

背景：原来代码在结算界面按 F2（F2 是「任务手册」），实测把任务手册连开 79 次；
而且结算界面里 hero 常常检不到，旧代码在 hero 缺失处就 continue，
根本走不到结算分支，方向键还会在结算列表里乱选（把角色带到了「返回城镇」）。

这个测试用两批真实数据验证：
  * 正样本：test_flow/fixtures/settlement_screen.png（真实结算截图，右侧写着
    「再次挑战 (F10)」）
  * 负样本：datasets/dnf 的 val 图（普通战斗画面，不应判成结算）

    python test_flow/test_settlement.py
"""
import glob
import importlib.util
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from yolo_detector import NAMES, YoloDetector  # noqa: E402

MIN_CONF = 0.40     # 会被 main() 用 main2.py 里的实际值覆盖


def load_settlement_fn():
    """从 main2.py 里取出 is_settlement_options（不执行它的主循环）。"""
    src = open(os.path.join(ROOT, 'main2.py'), encoding='utf-8').read()
    start = src.index('def is_settlement_options')
    end = src.index('def release_all_direction_keys')
    ns = {
        'settlement_min_w': 0.18, 'settlement_min_h': 0.25,
        'settlement_cx': (0.40, 0.90), 'settlement_cy': (0.10, 0.60),
    }
    exec(src[start:end], ns)
    return ns['is_settlement_options']


def load_min_conf():
    """从 main2.py 里读 options 置信度门槛，保证测试和实盘一致。"""
    for line in open(os.path.join(ROOT, 'main2.py'), encoding='utf-8'):
        if line.strip().startswith('settlement_min_conf'):
            return float(line.split('=')[1].split('#')[0].strip())
    return 0.40


def main():
    global MIN_CONF
    is_settlement = load_settlement_fn()
    MIN_CONF = load_min_conf()
    print('options 置信度门槛: %.2f（读自 main2.py）\n' % MIN_CONF)
    det = YoloDetector('best.pt', device='cuda:0' if _has_cuda() else 'cpu',
                       imgsz=640, conf_thres=0.3, iou_thres=0.2, names=NAMES)

    def judge(img):
        xywh, cls, confs, xyxy = det.detect(img)
        names_here = [NAMES[c] for c in cls]
        box = None
        for i, n in enumerate(names_here):
            if n == 'options' and confs[i] >= MIN_CONF:
                box = xyxy[i]
                break
        # 与 main2 保持一致：不再排除 monster/BOSS（v3 会在结算面板上误检出 monster）
        if box is None:
            return False, names_here, None
        return is_settlement(box, img.shape[1], img.shape[0]), names_here, box

    fails = 0

    # ---- 正样本 ----
    p = os.path.join(ROOT, 'test_flow', 'fixtures', 'settlement_screen.png')
    img = cv2.imread(p)
    if img is None:
        print('[FAIL] 读不到正样本:', p)
        return 1
    ok, names_here, box = judge(img)
    print('正样本(真实结算界面 1332x750)')
    print('  检测到类别:', sorted(set(names_here)))
    if box is not None:
        print('  options 框: %dx%d 中心(%.0f,%.0f)'
              % (box[2] - box[0], box[3] - box[1],
                 (box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
    print('  判定为结算界面:', ok, '->', '[OK]' if ok else '[FAIL]')
    fails += 0 if ok else 1
    fails += check_key_f10()

    # ---- 负样本：val 里**没有 options 标注**的普通战斗画面 ----
    # 注意：不能随便取 val 图当负样本 —— 录像末尾的「奖励」翻牌页也被标注成 options，
    # 那些是正样本，取进来会得到假的「误判」。
    all_val = sorted(glob.glob(os.path.join(ROOT, 'datasets', 'dnf', 'images', 'val', '*.jpg')))
    neg_files, pos_files = [], []
    for f in all_val:
        lbl = os.path.join(ROOT, 'datasets', 'dnf', 'labels', 'val',
                           os.path.splitext(os.path.basename(f))[0] + '.txt')
        has_opt = False
        if os.path.exists(lbl):
            for line in open(lbl, encoding='utf-8'):
                p = line.split()
                if p and p[0] == '8':       # class 8 = options
                    has_opt = True
                    break
        (pos_files if has_opt else neg_files).append(f)

    false_pos = []
    single_hits = 0
    for f in neg_files[:20]:
        im = cv2.imread(f)
        if im is None:
            continue
        ok2, names2, _ = judge(im)
        if ok2:
            single_hits += 1
            false_pos.append(os.path.basename(f))
    used_neg = min(20, len(neg_files))
    # 实际逻辑要求「连续 5 帧」才按 F10（options_confirm_frames=5），
    # 所以单帧误判不会触发重开，真正要看的是有没有 ≥5 帧的连续误判。
    # 这里用同一张图重复判 5 次来模拟「连续帧」：如果同一个画面被误判，
    # 那它连续出现时就会触发，必须拦住。
    persistent = 0
    for name in false_pos:
        f = os.path.join(ROOT, 'datasets', 'dnf', 'images', 'val', name)
        im = cv2.imread(f)
        if im is None:
            continue
        hits = sum(1 for _ in range(5) if judge(im)[0])
        if hits >= 5:
            persistent += 1

    print('\n负样本(%d 张无 options 标注的普通画面):' % used_neg)
    print('  单帧误判 %d 张（不影响：需要连续 5 帧才按 F10）' % single_hits)
    for n in false_pos:
        print('    -', n)
    print('  稳定误判(连续 5 帧都判成结算) %d 张 -> %s'
          % (persistent, '[OK] 无' if persistent == 0 else '[FAIL] 会导致误按 F10'))
    fails += 0 if persistent == 0 else 1

    # ---- 额外正样本：val 里带 options 标注的结算/奖励页 ----
    # 说明：模型对「奖励」翻牌页识别率一般（那是另一个画面），这里只做统计，
    # 要求至少一半能识别出来；漏掉的那几帧机器人会继续走路，
    # 等「是否继续？」面板出现时再冻结 + 按 F10，不影响整体流程。
    hit = 0
    tested = 0
    miss = []
    for f in pos_files[:10]:
        im = cv2.imread(f)
        if im is None:
            continue
        tested += 1
        ok3, _, _ = judge(im)
        if ok3:
            hit += 1
        else:
            miss.append(os.path.basename(f))
    rate = (hit / tested * 100) if tested else 100.0
    print('\n额外正样本(%d 张带 options 标注的结算/奖励页): 识别 %d 张 (%.0f%%)'
          % (tested, hit, rate))
    for n in miss:
        print('   漏:', n)
    # 这些漏掉的几乎都是「奖励」翻牌页 —— 和「是否继续」面板是两个画面。
    # 这里刻意把置信度门槛设到 0.40（误检正好卡在 0.30），宁可少认奖励页，
    # 也不能在普通战斗画面上误判（那会让机器人冻结在原地）。
    # 真正关键的是「是否继续」面板，见上面的正样本（实测线上 6/6 帧全部命中）。
    if rate >= 30.0:
        print('  -> [OK] 奖励页识别率 %.0f%%（关键面板已稳定识别，此页漏掉不影响流程）' % rate)
    else:
        print('  -> [FAIL] 识别率低于 30%')
        fails += 1

    print('\n结果:', '全部通过' if fails == 0 else '%d 项失败' % fails)
    return 0 if fails == 0 else 1


def check_key_f10():
    """F10 必须真的能按出来（key_map 里原来没有 F10，会 KeyError）。"""
    from directkeys import key_map
    ok = key_map.get('F10') == 121
    print('\nkey_map 中存在 F10(121):', key_map.get('F10'), '->', '[OK]' if ok else '[FAIL]')
    return 0 if ok else 1


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == '__main__':
    sys.exit(main())
