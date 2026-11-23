# test_flow —— 训练成果的离线验证

不依赖游戏窗口、不发送任何按键，用来确认 `best.pt` 能用、主循环能跑通。

## 一键跑（推荐）

```powershell
cd D:\Project\yolov5-DNF
.\test_flow\run.bat
```

它自己找装了 `torch + ultralytics + opencv` 的 Python，设好 UTF-8，依次跑全部检查。

只跑某一项：

```powershell
.\test_flow\run.bat labels      # 数据集每类实例数
.\test_flow\run.bat val         # 验证集 mAP / 每类 P/R
.\test_flow\run.bat sim         # 主循环模拟，默认 50 帧
.\test_flow\run.bat sim 200     # 跑 200 帧
.\test_flow\run.bat live        # 实时抓屏 + 检测链路检查
```

指定解释器：

```powershell
$env:DNF_PYTHON = 'D:\Project\yolov5-DNF\.venv\Scripts\python.exe'
.\test_flow\run.bat
```

## 直接跑单个脚本

工作目录必须是**项目根目录**（脚本都按项目根找 `best.pt` / `datasets` / `runs`）：

```powershell
cd D:\Project\yolov5-DNF
$env:PYTHONIOENCODING = 'utf-8'          # 不加的话中文可能乱码

E:\Model\yolo_cropDisease_detection_web\yolo_cropDisease_detection_flask\.venv\Scripts\python.exe `
  test_flow\val_metrics.py
```

## 脚本清单

| 脚本 | 作用 | 是否会操作游戏 |
|------|------|----------------|
| `val_metrics.py` | 在 `datasets/dnf` 的 val 上跑官方评估 | 否 |
| `simulate_main2.py` | 伪造 `grab_screen` + 拦截 `SendInput`/`imshow`，原样 exec `main2.py` 主循环 | 否（按键只计数，不发送） |
| `live_capture_check.py` | 对**整个桌面**某区域真抓屏 + 真推理 | 否 |
| `probe_game_window.py` | 自动定位游戏窗口客户区，抓一帧看实机识别效果并存图 | 否 |
| `label_stats.py` | 统计 train/val 每类实例数 | 否 |
| `test_skill_bar.py` | 技能栏轮转/冷却判定/出关节拍的离线单测（合成图，不碰游戏） | 否 |
| `test_settlement.py` | 结算界面识别 + F10 再次挑战的回归测试（用真实截图 + val 图正负样本） | 否 |
| `verify_skills.py` | 实机验证技能轮转与 X 回退（城镇里放技能无害） | 会按技能键 |
| `verify_skill_timing.py` | 实机验证 4 秒技能 + 第 2 秒 X 的节拍 | 会按技能键 |
| `minimap_look.py` | 把小地图区域裁出来放大，标定寻路用 | 否 |
| `label_geometry.py` | 统计各分类标注的位置/尺寸分布（用来给误检加几何约束） | 否 |
| `record_game.py` / `.bat` | 录制游戏客户区（1332x750，用于标注），倒计时后自动开录 | 只录屏 |
| `ls_import_frames.py` | 把一帧目录导入 Label Studio（走 Django ORM，不需要 API token） | 否 |

## 导入 Label Studio（不依赖 API token）

本机 `E:\Work\LabelStudio\api-token.txt` 里那个 token 在 Label Studio 1.23 已失效
（legacy token 被移除，`Bearer`/`Token` 都返回 401）。所以导入走 `label-studio shell`
（Django ORM），和 `export-yolo.ps1` 是同一套机制：

```powershell
# 新建项目导入
python test_flow/ls_import_frames.py --frames E:\Work\LabelStudio\data\media\frames\doors01 `
    --title "DNF-YOLO doors01（门/出口）"

# 追加到已有项目（可重复执行，按文件名去重）
python test_flow/ls_import_frames.py --frames <帧目录> --project-id 6
```

它会把帧复制到 `data\media\upload\<项目id>\<hash>-<文件名>`，任务里写
`/data/upload/<项目id>/...` —— 和现有任务完全一致的形式，网页里能直接显示。

> 导出是**累加**的：`export-yolo.ps1` 只 `mkdir` + `copy2`，从不删除。
> 所以单独导一个新项目会往 `datasets/dnf` 追加，不会覆盖原有图片。

## 结算界面与「再次挑战」

结算面板「是否继续？」里的真实快捷键是：

```
再次挑战        (F10)
快速组队        (未指定)
选择其它地下城  (F11)
返回城镇        (F12)
```

原代码按的是 **F2（任务手册）** —— 一按就弹窗挡画面，实测 60 秒连按 79 次。现在改成 `restart_key = 'F10'`，并且：

- 结算判定优先级最高，**放在 hero 判断之前**（结算界面里 hero 常常检不到，旧代码在
  hero 缺失处就 `continue`，永远走不到结算分支）；
- 判定到结算就**立刻冻结所有方向键**（面板是个列表，方向键会移动选中项，
  实测把机器人带到了「返回城镇」，直接退回城镇地图）；
- 三重过滤防误判：几何约束（训练标注统计）+ 连续 5 帧确认 + 有怪不算；
- 结算分「奖励」和「是否继续」两页，F10 只在后者有效，所以期间**每 2.5 秒重按一次**。

```powershell
python test_flow/test_settlement.py    # 回归测试：真实结算截图 + 20 张普通画面
```

> 已知：模型对「奖励」翻牌页识别率约 40%（那是另一个画面）。漏掉时机器人会继续走路，
> 等「是否继续」面板出现再冻结 + F10，不影响流程。误判率是 0 —— 这点更重要，
> 因为在普通画面上误判会让机器人冻结不动。

## 技能释放（skill_bar.py）

`main2.py` 打怪时不再固定按 `A`，改成按节拍出招：

- **每 4 秒放一个技能**，12 个技能 **A S D F G H Q W E R T Y** 按顺序轮流；
- 每个节拍**第 2 秒补一次 X**（普通攻击）；
- 每个技能格读亮度判断是否在冷却（图标变灰 = 亮度掉到基准的 ~50~65%），冷却中的跳过；
- 节拍起点如果没有可用技能，就按 X 顶上，这一轮不再补 X。

一个周期的样子：

```
t=0s  技能 A
t=2s  X
t=4s  技能 S
t=6s  X
t=8s  技能 D
...
```

参数在 `main2.py` 顶部：

| 参数 | 默认 | 说明 |
|---|---|---|
| `skill_rotation` | `A S D F G H Q W E R T Y` | 轮转顺序，改这里就能调整优先级 |
| `skill_interval` | `4.0` | 技能节拍（秒） |
| `skill_x_offset` | `2.0` | 一个节拍里第几秒补 X |
| `skill_ready_ratio` | `0.82` | 亮度 / 基准低于此值 → 认为冷却中 |
| `skill_repeat_lockout` | `0.35` | 同一技能这段时间内不重复按（秒） |

验证脚本：

```powershell
python test_flow/test_skill_bar.py        # 离线单测（含节拍时间线）
python test_flow/verify_skill_timing.py   # 实机 14 秒节拍验证（会真的按键）
python test_flow/verify_skills.py         # 实机验证轮转 + 冷却跳过
```

> 技能格坐标是按 **1332x750** 客户区实测标定的（`skill_bar.py` 里的 `SLOT_X0` 等）。
> 换分辨率/换 UI 缩放要重新量：`python skill_bar.py` 会存一张画了格子的
> `runs/test_flow/game/skillbar_ready.png`，对着图调坐标即可。

输出写到 `runs/test_flow/`：

- `val/`：ultralytics 评估产物
- `sim/`：主循环模拟中存下的带框图
- `live/`：实时抓屏那一帧的带框截图

## 真机运行（需要游戏窗口）

```powershell
cd D:\Project\yolov5-DNF
$py = 'E:\Model\yolo_cropDisease_detection_web\yolo_cropDisease_detection_flask\.venv\Scripts\python.exe'

# 0) 先确认抓屏区域对不对（不做任何操作）
& $py game_window.py
& $py test_flow\probe_game_window.py --activate

# 1) 先只看识别效果，不操作游戏
& $py yolo11_detect.py
#   运行中按 P 暂停/继续，按 q 退出

# 2) 确认识别正常后再自动挂机（会真的按键！）
& $py main2.py
```

> `main2.py` 会真的往当前窗口发按键。它启动时会先把游戏窗口切到前台，
> 切不过去就倒计时 5 秒让你自己点，还不行就直接 `SystemExit` 退出（不会误发按键）。

## 实机上踩过的三个坑（已在代码里处理）

1. **DPI 缩放**：显示器 1920x1080 @125% 时，Python 默认是「DPI 不感知」进程，
   `GetWindowRect` 会把 1332x750 的游戏窗口报成 1066x600，`grab_screen` 抓到的画面
   也被缩小 20%，和训练数据对不上，识别明显变差。
   → `grabscreen.py` 在 import 时就调用 `game_window.enable_dpi_awareness()`。

2. **抓屏区域写死成 `(0,0,1280,800)`**：游戏窗口根本不在左上角、尺寸也不是 1280x800。
   → `main2.py` / `yolo11_detect.py` 改成 `get_game_region()` 自动对准游戏客户区，
   找不到游戏窗口才退回旧值。你的机器上实测是 `(161,78)-(1493,828) = 1332x750`，
   正好等于数据集分辨率。

3. **`grab_screen` 抓的是桌面可见像素**：游戏被别的窗口挡住时抓到的就是挡它的窗口。
   → 所以游戏必须在前台。`main2.py` 启动时会自动切前台并校验。

## 注意

- `val_metrics.py` 的每类指标取自 `metrics.summary()`。不要用 `box.ap50[i]` 按类 id
  下标取——ultralytics 会把验证集里 0 实例的类别（本数据集是 `BOSS`）从数组里剔除，
  按下标取会整体错位。
- Windows 上 `workers` 保持 0，多进程 DataLoader 偶发 `worker exited unexpectedly`。
- `simulate_main2.py` 用 `ctypes.windll.user32.SendInput` 替身拦截按键，**不会**真的
  往当前窗口发按键；但它会修改进程内的 `ctypes`、`cv2`、`time.sleep`，只在测试进程内有效。
- `live_capture_check.py` 和 `yolo11_detect.py` 都依赖 pywin32。没有的话先
  `pip install pywin32`。
