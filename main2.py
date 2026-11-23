# -*- coding: utf-8 -*-
"""
DNF 自动挂机主循环 —— YOLOv11 (ultralytics) 版。

相对旧 main2.py 的改动：
  1. 用 ultralytics YOLOv11 替换 vendored YOLOv5 推理（见 yolo_detector.py）；
  2. 打怪 / 寻路 / 捡物 / 开箱 / 结算 的决策逻辑保持不变；
  3. 修正了进门判断里「门是否在左半边」用错维度（height→width）的 bug。

权重：项目根目录 best.pt（由 train.py 训练得到）。
"""

import random
import time

import cv2
import torch

import directkeys
from boss_bar import has_boss_bar
from direction_move import move
from directkeys import ReleaseKey
from game_window import (activate_game_window, describe_region, find_game_window,
                         get_game_region, is_game_foreground)
from getkeys import key_check
from grabscreen import grab_screen
from nav import (arrived, choose_door, door_aim_point, door_key, hero_foot, merge_doors, minimap_changed,
                 minimap_signature, walk_toward)
from skill_bar import SkillBar
from small_recgonize import current_door
from yolo_detector import YoloDetector, NAMES


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


# ==================== 参数 ====================
weights = r'best.pt'                    # YOLOv11 权重（train.py 训练得到）
# 抓屏区域：自动对准游戏窗口客户区（你的游戏窗口不一定在左上角、也不一定是 1280x800）。
# 找不到游戏窗口时退回旧的 (0,0,1280,800)。
window_size = get_game_region(fallback=(0, 0, 1279, 799))
print('抓屏区域:', describe_region(window_size))
img_size = 640                          # 推理输入尺寸
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
paused = False
view_img = True                         # 是否显示目标检测结果
conf_thres = 0.3                        # 置信度阈值
iou_thres = 0.2                         # NMS IoU 阈值
agnostic_nms = False
direct_dic = {"UP": 0xC8, "DOWN": 0xD0, "LEFT": 0xCB, "RIGHT": 0xCD}  # 方向键码
names = NAMES                           # ['hero','small_map','monster','money','material','door','BOSS','box','options']
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
action_cache = None                     # 当前按住的方向键标记
press_delay = 0.1                       # 按压时间
release_delay = 0.1                     # 释放时间
frame = 0                               # 帧计数
door1_time_start = -20                  # 第一门计时
fs = 1                                  # 每 fs 帧推理一次

# ---------- 防卡死参数 ----------
pickup_thresh = 45      # 捡东西的判定半径（像素）。原来是 15，实际走不到那么准，会原地打转
stuck_after = 15.0      # 超过这么多秒没有任何「进展」（攻击/捡物/进门/开箱）就算卡住
last_progress = time.time()   # 上次有进展的时间
last_recover = 0.0            # 上次脱困动作的时间
options_frames = 0            # 连续检测到结算界面的帧数
options_confirm_frames = 5    # 连续多少帧确认才算真的结算（过滤误检）
last_door_print = 0.0         # 「走到门口」日志节流
last_restart_time = 0.0       # 再次挑战的限速
settlement_hold_until = 0.0   # 结算粘滞：认到结算后，锁住到的时刻
settlement_hold_seconds = 4.0 # 粘滞时长（识别闪断时防止发方向键）
prev_mm_sig = None            # 上一帧小地图指纹（判断是否换房间）
last_door_target = None       # 当前正在走的目标门（用于判断「这扇门走不过去」）
same_door_since = 0.0         # 开始走这扇门的时间
ignore_doors_until = 0.0      # 在这之前先不理门，让脱困把角色挪开

# ---------- 清怪优先 / 门口卡住 ----------
# 模型会漏检怪（recall ~0.53，特效糊屏更低），单帧没看到不代表清完了。
# 只有「最近 monster_clear_confirm 秒都没见过怪」并且没有 Boss 血条，才允许进门。
monster_clear_confirm = 2.0   # 连续这么多秒没看到怪，才算房间清空（A 方案：宁可多等）
last_monster_time = -1e9      # 最近一次看到怪的时刻
door_arrived_since = 0.0      # 走到门口的时刻（用于判断「在门口挤太久」）
door_dwell_max = 3.0          # 站在门口超过这么久还没换房 → 开始上下摆动往里挤
door_wiggle_max = 6.0         # 挤这么久还不行 → 放弃这扇门，换一扇
door_target_max = 8.0         # 同一扇门走了这么久还没到 → 也放弃（够不着的假门）
bad_doors = {}                # 门 key -> 黑名单到期时刻（放弃过的门别再一头撞回去）

# ---------- 技能参数 ----------
# 轮转顺序：小技能优先，大招往后放。想改顺序直接改这里。
skill_rotation = ['A', 'S', 'D', 'F', 'G', 'H', 'Q', 'W', 'E', 'R', 'T', 'Y']
skill_ready_ratio = 0.82      # 技能格亮度 / 该格历史最亮 < 这个值 → 认为在冷却（图标变灰）
skill_min_interval = 0.10     # 两次按键最小间隔（秒）
skill_repeat_lockout = 0.35   # 同一技能这段时间内不重复按（秒）

# 出关节拍：每 4 秒放一个技能，中间的 2 秒补一次 X（普通攻击）
skill_interval = 4.0          # 技能节拍（秒）
skill_x_offset = 2.0          # 一个节拍里第几秒补 X

# ---------- 安全开关 ----------
# box 类误检很常见（蓝色灯柱会被认成 box），而 ESC 是系统菜单开关：
# 误按 ESC 会打开系统菜单，后续方向键+X 还会在菜单里乱选。
box_esc_enabled = False
# 结算界面「再次挑战」：界面里写的是 F10（不是 F2，F2 是任务手册）。
restart_enabled = True
restart_key = 'F10'
# 结算分「奖励」和「是否继续」两页，F10 只在后者有效，所以结算期间周期性重按
restart_period = 2.5
# 结算界面的几何约束（归一化，来自 datasets/dnf 里 56 个 options 标注的实测范围）
settlement_min_w = 0.18
settlement_min_h = 0.25
settlement_cx = (0.40, 0.90)
settlement_cy = (0.10, 0.60)
# options 置信度门槛：dnf_v3 实测「真结算面板」0.528、「易误判帧」最高 0.315，取 0.45。
# 宁可漏掉「奖励」翻牌页（漏了只是继续走路，等「是否继续」面板再按 F10），
# 也不能在普通画面上误判 —— 那会让机器人冻结并误按 F10。
settlement_min_conf = 0.45
# 游戏窗口不在前台时不发任何按键（否则按键会打到别的窗口上）
focus_guard_enabled = True

detector = YoloDetector(weights, device=device, imgsz=img_size,
                        conf_thres=conf_thres, iou_thres=iou_thres,
                        agnostic_nms=agnostic_nms, names=names)

# 技能轮转 + 冷却检测：12 个技能（A S D F G H / Q W E R T Y）按顺序轮流放，
# 图标变灰（冷却中）就跳过；全部不可用时改按 X（普通攻击）。
skill_bar = SkillBar(rotation=skill_rotation, ready_ratio=skill_ready_ratio,
                     min_interval=skill_min_interval,
                     repeat_lockout=skill_repeat_lockout,
                     skill_interval=skill_interval,
                     basic_x_offset=skill_x_offset)

# 启动前确认游戏在前台：按键是发给「当前前台窗口」的，
# 游戏不在前台就会把按键发到别的窗口上。切不过去就直接退出。
if not activate_game_window(countdown=5):
    raise SystemExit(1)

game_hwnd = find_game_window()
last_focus_warn = 0.0


def describe_action(key):
    """把按下的键说成人话：X 是普通攻击，其它是技能。"""
    if not key:
        return ''
    if key == 'X':
        return '普通攻击 X'
    return '释放技能 %s' % key


def is_settlement_options(box_xyxy, img_w, img_h):
    """判断一个 options 检测框像不像真的结算界面。

    阈值不是拍脑袋的，是统计 datasets/dnf 里 56 个 options 标注得到的：
        宽   255~1143（中位 492）
        高   204~ 713（中位 652）
        中心 x 658~1164，中心 y 136~394（画面 1332x750）
    误检那次是 199x219、中心 x=1233 —— 尺寸偏小且贴在最右边，正好被这几条挡掉。

    :param box_xyxy: [x1, y1, x2, y2]
    """
    if box_xyxy is None:
        return False
    x1, y1, x2, y2 = box_xyxy
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return (w >= settlement_min_w * img_w
            and h >= settlement_min_h * img_h
            and settlement_cx[0] * img_w <= cx <= settlement_cx[1] * img_w
            and settlement_cy[0] * img_h <= cy <= settlement_cy[1] * img_h)


def show_preview(img):
    """显示检测预览。

    预览窗口（标题 'window'）是唯一的人为可破坏点：用户点了它的关闭按钮之后，
    再 cv2.imshow 会抛 cv2.error，整条主循环就死了（实测进程无输出退出、exit 1）。
    这里吞掉这类错误，并且把 resize 结果放局部变量，不再覆盖调用方的 img。
    """
    if not view_img:
        return
    try:
        small = cv2.resize(img, (600, 375))
        cv2.imshow('window', small)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            raise StopIteration
    except cv2.error as exc:
        print("预览窗口异常，已忽略: %s" % exc)


def release_all_direction_keys():
    """松开可能按住的键：进程被切走/暂停时，避免角色一直朝一个方向跑。"""
    global action_cache
    if not action_cache:
        return
    if action_cache not in ["LEFT", "RIGHT", "UP", "DOWN"]:
        try:
            ReleaseKey(direct_dic[action_cache.strip().split("_")[0]])
            ReleaseKey(direct_dic[action_cache.strip().split("_")[1]])
        except Exception:
            pass
    else:
        try:
            ReleaseKey(direct_dic[action_cache])
        except Exception:
            pass
    action_cache = None


# ==================== 主循环：截屏 + 检测 + 决策 ====================
while True:
    if not paused:
        # 游戏不在前台就完全不动：按键会打到别的窗口上去
        if focus_guard_enabled and not is_game_foreground(game_hwnd):
            release_all_direction_keys()
            if time.time() - last_focus_warn > 5.0:
                last_focus_warn = time.time()
                print("游戏窗口不在前台，已暂停发按键（点回游戏窗口自动继续）")
            time.sleep(0.2)
            continue

        t_start = time.time()
        img0 = grab_screen(window_size)               # BGRA，供 current_door 使用
        frame += 1
        if frame % fs == 0:
            img = cv2.cvtColor(img0, cv2.COLOR_BGRA2BGR)   # BGR，供推理与显示
            xywh_list, cls_ids, confs, xyxy_list = detector.detect(img)
            # 左上角 Boss/精英血条：打 Boss 时满屏特效会把 Boss 和自己的框都糊掉，
            # 模型认不出来，但只要血条在就说明正在打，照放技能。
            boss_here = has_boss_bar(img)

            # ==================== 结算界面：优先级最高 ====================
            # 结算面板「是否继续？」里写着：再次挑战 (F10) / 选择其它地下城 (F11) / 返回城镇 (F12)。
            # 这张脸上 hero 经常检不到，所以必须放在 hero 判断之前处理，否则永远走不到。
            # 而且确认到之前那几帧也绝不能发方向键 —— 面板是个列表，方向键会移动选中项，
            # 实测把机器人带到了「返回城镇」，直接退回城镇地图。
            _opt_box = None
            for _i, _c in enumerate(cls_ids):
                if names[_c] == 'options' and confs[_i] >= settlement_min_conf:
                    _opt_box = xyxy_list[_i]
                    break
            settlement = (_opt_box is not None
                          and is_settlement_options(_opt_box, img.shape[1], img.shape[0]))
            if settlement:
                options_frames += 1
            else:
                options_frames = 0

            # 结算「粘滞」：识别会闪断（有的帧认到、有的没认到），闪断那几帧如果放行，
            # 机器人会按方向键 —— 而结算面板是个列表，方向键会移动选中项，
            # 有被带到「返回城镇」的风险。所以一旦认到就锁住一小段时间。
            if settlement or time.time() < settlement_hold_until:
                if settlement:
                    settlement_hold_until = time.time() + settlement_hold_seconds
                release_all_direction_keys()  # 结算界面里不发任何方向键
                # 结算有两个阶段：先是「奖励」翻牌页，然后才是「是否继续？」面板。
                # F10 只在后一阶段有效，所以在结算期间按周期重按，而不是只按一次。
                if restart_enabled and options_frames >= options_confirm_frames \
                        and time.time() - last_restart_time > restart_period:
                    last_restart_time = time.time()
                    directkeys.key_press(restart_key)
                    print("结算界面：按 %s 再次挑战" % restart_key)
                last_progress = time.time()
                show_preview(img)
                continue

            if len(xywh_list):
                img_object = []      # 每个目标的 [cx, cy, w, h]
                cls_object = []      # 每个目标的类别名
                hero_conf = 0
                hero_index = 0
                for idx, (xywh, cid, conf) in enumerate(zip(xywh_list, cls_ids, confs)):
                    img_object.append(xywh)
                    cls_object.append(names[cid])

                    if names[cid] == "hero" and conf > hero_conf:
                        hero_conf = conf
                        hero_index = idx

                    if view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[cid], conf)
                        plot_one_box(xyxy_list[idx], img, label=label, color=colors[cid], line_thickness=2)

                # ==================== 游戏决策 ====================
                thx = 30    # 捡东西/进门时 x 方向阈值
                thy = 30    # 捡东西/进门时 y 方向阈值
                attx = 150  # 攻击时 x 方向阈值
                atty = 50   # 攻击时 y 方向阈值

                # 用小地图画面变化判断「换房间了」：换房间就重置卡死计时，
                # 不然刚进门那一下会被当成「卡住」。
                mm_box = None
                for _i, _c in enumerate(cls_object):
                    if _c == 'small_map':
                        mm_box = xyxy_list[_i]
                        break
                cur_mm_sig = minimap_signature(img, mm_box)
                # 阈值给松一点：角色在小地图格子内移动也会让画面变一点，
                # 太灵敏会不断重置卡死计时，反而让「脱困」永远等不到。
                if minimap_changed(prev_mm_sig, cur_mm_sig, thresh=20.0):
                    print("小地图变化：换房间了")
                    last_progress = time.time()
                    action_cache = None
                    bad_doors.clear()          # 换房间了，上一间房的黑名单作废
                if cur_mm_sig is not None:
                    prev_mm_sig = cur_mm_sig

                # 第一房间：进入后按 CTRL+ALT 触发某机制
                if current_door(img0) == 1 and time.time() - door1_time_start > 10:
                    door1_time_start = time.time()
                    directkeys.key_press("CTRL")
                    time.sleep(1)
                    directkeys.key_press("ALT")
                    time.sleep(0.5)
                    action_cache = None

                # 扫描英雄
                if "hero" in cls_object:
                    hero_xywh = img_object[hero_index]
                    cv2.circle(img, (int(hero_xywh[0]), int(hero_xywh[1])), 1, (0, 0, 255), 10)
                else:
                    # 没检出 hero（多半是技能特效糊屏）。血条在就说明正在打 Boss，
                    # 直接放技能，别干等着。
                    if boss_here:
                        used = skill_bar.attack_tick(img0, directkeys.key_press)
                        if used:
                            print("Boss战(血条) %s（hero 未检出）" % describe_action(used))
                            last_progress = time.time()
                    continue

                # ---------- 打怪 ----------
                if "monster" in cls_object or "BOSS" in cls_object:
                    # 清怪优先（A 方案）：只要这一帧看到了怪，就记下来。
                    # 模型会漏检（monster recall 约 0.53，特效糊屏时更低），
                    # 单帧没看到不代表清完了，所以进门判断要参考「最近还见过怪」。
                    last_monster_time = time.time()
                    min_distance = float("inf")
                    for idx, (c, box) in enumerate(zip(cls_object, img_object)):
                        if c == 'monster' or c == "BOSS":
                            dis = ((hero_xywh[0] - box[0]) ** 2 + (hero_xywh[1] - box[1]) ** 2) ** 0.5
                            if dis < min_distance:
                                monster_box = box
                                monster_index = idx
                                min_distance = dis
                    # 处于攻击距离
                    if abs(hero_xywh[0] - monster_box[0]) < attx and abs(hero_xywh[1] - monster_box[1]) < atty:
                        # 轮流放可用技能；全在冷却就按 X 普通攻击。
                        # 用 img0（未画框的原始帧）读技能栏，避免检测框盖住图标影响判断。
                        used = skill_bar.attack_tick(img0, directkeys.key_press)
                        if used:
                            print(describe_action(used))
                        last_progress = time.time()
                        if not action_cache:
                            pass
                        elif action_cache not in ["LEFT", "RIGHT", "UP", "DOWN"]:
                            ReleaseKey(direct_dic[action_cache.strip().split("_")[0]])
                            ReleaseKey(direct_dic[action_cache.strip().split("_")[1]])
                            action_cache = None
                        elif action_cache:
                            ReleaseKey(direct_dic[action_cache])
                            action_cache = None
                    # 怪物在英雄右上方
                    elif monster_box[1] - hero_xywh[1] < 0 and monster_box[0] - hero_xywh[0] > 0:
                        if abs(monster_box[1] - hero_xywh[1]) < thy:
                            action_cache = move(direct="RIGHT", material=True, action_cache=action_cache,
                                                press_delay=press_delay, release_delay=release_delay)
                        elif hero_xywh[1] - monster_box[1] < monster_box[0] - hero_xywh[0]:
                            action_cache = move(direct="RIGHT_UP", material=True, action_cache=action_cache,
                                                press_delay=press_delay, release_delay=release_delay)
                        elif hero_xywh[1] - monster_box[1] >= monster_box[0] - hero_xywh[0]:
                            action_cache = move(direct="UP", material=True, action_cache=action_cache,
                                                press_delay=press_delay, release_delay=release_delay)
                    # 怪物在英雄左上方
                    elif monster_box[1] - hero_xywh[1] < 0 and monster_box[0] - hero_xywh[0] < 0:
                        if abs(monster_box[1] - hero_xywh[1]) < thy:
                            action_cache = move(direct="LEFT", material=True, action_cache=action_cache,
                                                press_delay=press_delay, release_delay=release_delay)
                        elif hero_xywh[1] - monster_box[1] < hero_xywh[0] - monster_box[0]:
                            action_cache = move(direct="LEFT_UP", material=True, action_cache=action_cache,
                                                press_delay=press_delay, release_delay=release_delay)
                        elif hero_xywh[1] - monster_box[1] >= hero_xywh[0] - monster_box[0]:
                            action_cache = move(direct="UP", material=True, action_cache=action_cache,
                                                press_delay=press_delay, release_delay=release_delay)
                    # 怪物在英雄左下方
                    elif monster_box[1] - hero_xywh[1] > 0 and monster_box[0] - hero_xywh[0] < 0:
                        if abs(monster_box[1] - hero_xywh[1]) < thy:
                            action_cache = move(direct="LEFT", material=True, action_cache=action_cache,
                                                press_delay=press_delay, release_delay=release_delay)
                        elif monster_box[1] - hero_xywh[1] < hero_xywh[0] - monster_box[0]:
                            action_cache = move(direct="LEFT_DOWN", material=True, action_cache=action_cache,
                                                press_delay=press_delay, release_delay=release_delay)
                        elif monster_box[1] - hero_xywh[1] >= hero_xywh[0] - monster_box[0]:
                            action_cache = move(direct="DOWN", material=True, action_cache=action_cache,
                                                press_delay=press_delay, release_delay=release_delay)
                    # 怪物在英雄右下方
                    elif monster_box[1] - hero_xywh[1] > 0 and monster_box[0] - hero_xywh[0] > 0:
                        if abs(monster_box[1] - hero_xywh[1]) < thy:
                            action_cache = move(direct="RIGHT", material=True, action_cache=action_cache,
                                                press_delay=press_delay, release_delay=release_delay)
                        elif monster_box[1] - hero_xywh[1] < monster_box[0] - hero_xywh[0]:
                            action_cache = move(direct="RIGHT_DOWN", material=True, action_cache=action_cache,
                                                press_delay=press_delay, release_delay=release_delay)
                        elif monster_box[1] - hero_xywh[1] >= monster_box[0] - hero_xywh[0]:
                            action_cache = move(direct="DOWN", material=True, action_cache=action_cache,
                                                press_delay=press_delay, release_delay=release_delay)

                # ---------- 移动到下一个地图（进门） ----------
                # 原逻辑：地面上还有 money/material 就不进门（否则捡不起来会死循环）。
                # 现在叠加「清怪优先」：最近 monster_clear_confirm 秒内见过怪、或者左上
                # Boss 血条还在，都不许进门 —— 模型漏检时不能当成房间已清。
                _has_drop = ("material" in cls_object) or ("money" in cls_object)
                _stuck = (time.time() - last_progress) > stuck_after
                _mobs_clear = (time.time() - last_monster_time) > monster_clear_confirm
                if "door" in cls_object and "monster" not in cls_object and "BOSS" not in cls_object \
                        and _mobs_clear and not boss_here \
                        and (not _has_drop or _stuck) and time.time() >= ignore_doors_until:
                    # 模型对同一个门口常出 2~3 个框（两根灯柱各一个），原来取「列表最后一个」
                    # 也就是置信度最低的那个，方向就飘了。这里先合并成一个门，再选目标走过去。
                    door_boxes = [b for c, b in zip(cls_object, img_object) if c == 'door']
                    merged_doors = merge_doors(door_boxes)
                    # 过滤掉最近判定「走不过去」的门：模型会把墙边的灯柱认成门，
                    # 不拉黑就会一直往同一根柱子上撞（实测 39 秒原地不动）。
                    _now = time.time()
                    merged_doors = [d for d in merged_doors
                                    if bad_doors.get(door_key(d), 0.0) < _now]
                    hero_pt = hero_foot(hero_xywh)
                    target = choose_door(merged_doors, hero_pt, img.shape[1], allow_left=_stuck)
                    if target is not None:
                        _tx, _ty, door_rect = target
                        # 瞄准「离角色最近的门框边」再往里缩一点，而不是几何中心。
                        # 左右两侧的门框常被墙/柱子切掉一半，中心点可能落在墙里或屏幕外，
                        # 照中心走就会在门口一直顶着墙跑（实测左右门卡死）。
                        aim_x, aim_y = door_aim_point(hero_pt, door_rect)

                        tgt = door_key(door_rect)
                        if tgt != last_door_target:
                            last_door_target = tgt
                            same_door_since = time.time()
                            door_arrived_since = 0.0
                        elif time.time() - same_door_since > door_target_max and _stuck:
                            # 够不着的假门：走了 door_target_max 秒还没到，拉黑一会儿换一扇
                            bad_doors[tgt] = time.time() + 20.0
                            last_door_target = None
                            door_arrived_since = 0.0
                            print("这扇门走了 %.0f 秒都没走到，换一扇" % (time.time() - same_door_since))
                            continue

                        swing = 0
                        if arrived(hero_pt, door_rect):
                            # 只在「刚到达」时记一次进展。之前这里每帧都刷新 last_progress，
                            # 导致 _stuck 永远不成立 → 放弃逻辑永远触发不了 → 门口死锁。
                            if door_arrived_since == 0.0:
                                door_arrived_since = time.time()
                                last_progress = time.time()
                            dwell = time.time() - door_arrived_since
                            if dwell > door_wiggle_max:
                                bad_doors[tgt] = time.time() + 20.0
                                ignore_doors_until = time.time() + 3.0
                                last_door_target = None
                                door_arrived_since = 0.0
                                print("门口挤了 %.0f 秒还没换房间，放弃这扇门" % dwell)
                            elif dwell > door_dwell_max:
                                # 门洞可能不在正中：上下摆动地往里挤
                                swing = 45 if int(dwell * 2) % 2 == 0 else -45
                                if time.time() - last_door_print > 1.5:
                                    last_door_print = time.time()
                                    print("在门口挤一挤（%.1f 秒）" % dwell)
                        else:
                            door_arrived_since = 0.0
                            if time.time() - last_door_print > 3.0:
                                last_door_print = time.time()
                                print("走向门口")

                        d = walk_toward(hero_pt, (aim_x, aim_y + swing), dead_zone=15)
                        if d:
                            action_cache = move(direct=d, action_cache=action_cache,
                                                press_delay=press_delay, release_delay=release_delay)

                # ---------- 无任何目标：默认向右走 ----------
                if "money" not in cls_object and "material" not in cls_object and "monster" not in cls_object \
                        and "BOSS" not in cls_object and "door" not in cls_object and 'box' not in cls_object \
                        and 'options' not in cls_object:
                    action_cache = move(direct="RIGHT", action_cache=action_cache,
                                        press_delay=press_delay, release_delay=release_delay)

                # ---------- 捡材料 / 金币 ----------
                # 注意：卡住且画面里有门时，让「进门」优先，否则进门和捡物两块每帧互相覆盖，
                # 角色就会原地左右横跳（实测就是这个死循环）。
                _defer_pickup = _stuck and ("door" in cls_object)
                if "monster" not in cls_object and "hero" in cls_object and (
                        "material" in cls_object or "money" in cls_object) and not _defer_pickup:
                    min_distance = float("inf")
                    hero_xywh[1] = hero_xywh[1] + (hero_xywh[3] // 2) * 0.7  # 英雄框为上半身，中心下移
                    # 原来把阈值砍半成 15 像素，实际根本走不到那么准，会一直绕圈捡不到。
                    thx = pickup_thresh
                    thy = pickup_thresh
                    for idx, (c, box) in enumerate(zip(cls_object, img_object)):
                        if c == 'material' or c == "money":
                            dis = ((hero_xywh[0] - box[0]) ** 2 + (hero_xywh[1] - box[1]) ** 2) ** 0.5
                            if dis < min_distance:
                                material_box = box
                                material_index = idx
                                min_distance = dis
                    if abs(material_box[1] - hero_xywh[1]) < thy and abs(material_box[0] - hero_xywh[0]) < thx:
                        if not action_cache:
                            pass
                        elif action_cache not in ["LEFT", "RIGHT", "UP", "DOWN"]:
                            ReleaseKey(direct_dic[action_cache.strip().split("_")[0]])
                            ReleaseKey(direct_dic[action_cache.strip().split("_")[1]])
                            action_cache = None
                        else:
                            ReleaseKey(direct_dic[action_cache])
                            action_cache = None
                        time.sleep(1)
                        directkeys.key_press("X")
                        print("捡东西")
                        last_progress = time.time()
                    elif material_box[1] - hero_xywh[1] < 0 and material_box[0] - hero_xywh[0] > 0:
                        if abs(material_box[1] - hero_xywh[1]) < thy:
                            action_cache = move(direct="RIGHT", material=True, action_cache=action_cache,
                                                press_delay=press_delay, release_delay=release_delay)
                        elif hero_xywh[1] - material_box[1] < material_box[0] - hero_xywh[0]:
                            action_cache = move(direct="RIGHT_UP", material=True, action_cache=action_cache,
                                                press_delay=press_delay, release_delay=release_delay)
                        elif hero_xywh[1] - material_box[1] >= material_box[0] - hero_xywh[0]:
                            action_cache = move(direct="UP", material=True, action_cache=action_cache,
                                                press_delay=press_delay, release_delay=release_delay)
                    elif material_box[1] - hero_xywh[1] < 0 and material_box[0] - hero_xywh[0] < 0:
                        if abs(material_box[1] - hero_xywh[1]) < thy:
                            action_cache = move(direct="LEFT", material=True, action_cache=action_cache,
                                                press_delay=press_delay, release_delay=release_delay)
                        elif hero_xywh[1] - material_box[1] < hero_xywh[0] - material_box[0]:
                            action_cache = move(direct="LEFT_UP", material=True, action_cache=action_cache,
                                                press_delay=press_delay, release_delay=release_delay)
                        elif hero_xywh[1] - material_box[1] >= hero_xywh[0] - material_box[0]:
                            action_cache = move(direct="UP", material=True, action_cache=action_cache,
                                                press_delay=press_delay, release_delay=release_delay)
                    elif material_box[1] - hero_xywh[1] > 0 and material_box[0] - hero_xywh[0] < 0:
                        if abs(material_box[1] - hero_xywh[1]) < thy:
                            action_cache = move(direct="LEFT", material=True, action_cache=action_cache,
                                                press_delay=press_delay, release_delay=release_delay)
                        elif material_box[1] - hero_xywh[1] < hero_xywh[0] - material_box[0]:
                            action_cache = move(direct="LEFT_DOWN", material=True, action_cache=action_cache,
                                                press_delay=press_delay, release_delay=release_delay)
                        elif material_box[1] - hero_xywh[1] >= hero_xywh[0] - material_box[0]:
                            action_cache = move(direct="DOWN", material=True, action_cache=action_cache,
                                                press_delay=press_delay, release_delay=release_delay)
                    elif material_box[1] - hero_xywh[1] > 0 and material_box[0] - hero_xywh[0] > 0:
                        if abs(material_box[1] - hero_xywh[1]) < thy:
                            action_cache = move(direct="RIGHT", material=True, action_cache=action_cache,
                                                press_delay=press_delay, release_delay=release_delay)
                        elif material_box[1] - hero_xywh[1] < material_box[0] - hero_xywh[0]:
                            action_cache = move(direct="RIGHT_DOWN", material=True, action_cache=action_cache,
                                                press_delay=press_delay, release_delay=release_delay)
                        elif material_box[1] - hero_xywh[1] >= material_box[0] - hero_xywh[0]:
                            action_cache = move(direct="DOWN", material=True, action_cache=action_cache,
                                                press_delay=press_delay, release_delay=release_delay)

                # ---------- 开箱子 ----------
                # 注意：ESC 在 DNF 里是「系统菜单」开关。box 类经常误检（蓝色灯柱会被
                # 认成 box），实测会连按 ESC 把系统菜单打开，接着方向键+X 还会在菜单里
                # 乱选（可能误点「返回城镇」/「结束游戏」）。所以默认关闭。
                if box_esc_enabled and "box" in cls_object:
                    box_num = 0
                    for b in cls_object:
                        if b == "box":
                            box_num += 1
                    if box_num >= 4:
                        directkeys.key_press("ESC")
                        print("打开箱子ESC")
                        last_progress = time.time()

                # ---------- Boss 血条兜底 ----------
                # 打 Boss 时满屏技能特效会把 Boss 和自己的检测框都糊掉（实测 hero 0.07、
                # monster 0.17），决策层「看不到怪」就会乱走。只要左上血条还在，
                # 说明正在打，直接照放技能 / 普通攻击。
                if boss_here and "monster" not in cls_object and "BOSS" not in cls_object:
                    used = skill_bar.attack_tick(img0, directkeys.key_press)
                    if used:
                        print("Boss战(血条) %s" % describe_action(used))
                    last_progress = time.time()

                # ---------- 脱困：太久没有进展就盲按 X + 换方向 ----------
                # 场景：地上的东西一直走不到（阈值/遮挡/贴图重叠），或卡在门口判定不到。
                # 不处理的话角色会原地左右横跳，永远出不来。
                if (time.time() - last_progress) > stuck_after \
                        and (time.time() - last_recover) > 2.0:
                    last_recover = time.time()
                    if action_cache:
                        if action_cache not in ["LEFT", "RIGHT", "UP", "DOWN"]:
                            ReleaseKey(direct_dic[action_cache.strip().split("_")[0]])
                            ReleaseKey(direct_dic[action_cache.strip().split("_")[1]])
                        else:
                            ReleaseKey(direct_dic[action_cache])
                        action_cache = None
                    directkeys.key_press("X")
                    random_dir = random.choice(["LEFT", "RIGHT", "UP", "DOWN"])
                    action_cache = move(direct=random_dir, action_cache=action_cache,
                                        press_delay=press_delay, release_delay=release_delay)
                    print("脱困：盲按 X + 换方向 %s（已 %.0f 秒无进展）"
                          % (random_dir, time.time() - last_progress))

            else:
                # 一帧什么都没检到（换场景 / 读图 / 城镇等训练分布外的画面）。
                if boss_here:
                    # 但血条在：说明正在打 Boss，只是特效糊住了整个画面
                    used = skill_bar.attack_tick(img0, directkeys.key_press)
                    if used:
                        print("Boss战(血条) %s" % describe_action(used))
                    last_progress = time.time()
                elif (time.time() - last_recover) > 1.0:
                    # 原来这里直接什么都不做，角色会一直发呆；给个默认向右走。
                    last_recover = time.time()
                    action_cache = move(direct="RIGHT", action_cache=action_cache,
                                        press_delay=press_delay, release_delay=release_delay)
                    print("未检出任何目标，默认向右走")

            t_end = time.time()
            print("一帧游戏操作所用时间：", (t_end - t_start) / fs)

            show_preview(img)

    # Setting pause and unpause
    keys = key_check()
    if 'P' in keys:
        if not action_cache:
            pass
        elif action_cache not in ["LEFT", "RIGHT", "UP", "DOWN"]:
            ReleaseKey(direct_dic[action_cache.strip().split("_")[0]])
            ReleaseKey(direct_dic[action_cache.strip().split("_")[1]])
            action_cache = None
        else:
            ReleaseKey(direct_dic[action_cache])
            action_cache = None
        if paused:
            paused = False
            time.sleep(1)
        else:
            paused = True
            time.sleep(1)
