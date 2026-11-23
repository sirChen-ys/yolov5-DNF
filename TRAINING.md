# DNF YOLOv11 训练与使用说明

本项目已从内置的 vendored YOLOv5 升级为 **ultralytics YOLOv11**，并补齐了训练管线。
类别沿用原项目的 **9 类**：`hero / small_map / monster / money / material / door / BOSS / box / options`。

---

## 一、目录说明（新增/改动）

| 文件 | 作用 |
|------|------|
| `yolo_detector.py` | YOLOv11 推理封装，输出与旧决策逻辑兼容的检测结果 |
| `main2.py` | 主循环（重写推理部分，决策逻辑不变） |
| `yolo11_detect.py` | 纯检测演示（替代旧 `yolo5_detect.py`） |
| `train.py` | 训练脚本 |
| `data/dnf.yaml` | 数据集与类别配置 |
| `prepare_dataset.py` | **[已废弃]** labelme 标注 → YOLO 数据集（仅用于转换历史数据） |
| `requirements.txt` | 依赖清单 |
| `models/`、`utils/` | **旧 YOLOv5 代码，已不再被主流程引用**（保留供 test/ 旧脚本使用） |

---

## 二、安装依赖

```bash
pip install -r requirements.txt
# 或最小安装
pip install torch ultralytics opencv-python numpy pywin32
```

> `pywin32`、`pyautogui`、`pydirectinput` 仅 Windows 截屏/键鼠需要。
> 首次训练时 ultralytics 会自动下载预训练权重（如 `yolo11n.pt`）。

---

## 三、准备数据集

标注工具为 **Label Studio**（本地服务 http://localhost:8080 ，项目「DNF-YOLO验证｜目标检测」）。
旧的 labelme 流程已废弃，见第六节。

### 1. 抽帧
用 `E:\Work\LabelStudio\record-screen.ps1` 录屏，再用 `extract-frames.ps1` 抽帧：

```powershell
E:\Work\LabelStudio\extract-frames.ps1 `
  -Video E:\GameData\raw\match01_game.mp4 `
  -OutDir E:\Work\LabelStudio\data\media\frames\match01 -Fps 2
```

### 2. 在 Label Studio 中标注
浏览器打开 http://localhost:8080/projects/4 ，按项目说明里的规范标注。

**类别（9 类，英文，必须与 `data/dnf.yaml` 完全一致）：**

```
ID 0: hero        ID 3: money       ID 6: BOSS
ID 1: small_map   ID 4: material    ID 7: box
ID 2: monster     ID 5: door        ID 8: options
```

> 注意：`money`（金币）与 `material`（材料）**必须分开标注**，
> `main2.py` 的决策逻辑依赖区分二者。

### 3. 导出为 YOLO 数据集

```powershell
E:\Work\LabelStudio\export-yolo.ps1 -Project 4 -SplitByGroup
```

导出脚本会自动：
- 从 `data/dnf.yaml` 读取类别顺序（**不按字母序重排**，保证 class_id 与训练/推理一致）
- 输出到 `D:\Project\yolov5-DNF\datasets\dnf`
- 同时生成 `classes.txt` 与 `data.yaml`（`nc: 9`）

输出结构（ultralytics 标准格式）：

```
datasets/dnf/
├── images/
│   ├── train/*.jpg
│   └── val/*.jpg
├── labels/
│   ├── train/*.txt    # 每行: class_id cx cy w h（归一化）
│   └── val/*.txt
├── classes.txt
└── data.yaml
```

> `-SplitByGroup` 会**按视频分组**切分 train/val。务必加上：
> 同一段录像的相邻帧几乎一样，若按帧随机切分，训练集与验证集会互相泄漏，
> 导致 mAP 虚高、上线失效。

### 4. 确认配置
`data/dnf.yaml` 的 `path: datasets/dnf` 是相对项目根的，
默认导出位置与之匹配，**无需修改**。

---

## 四、训练

```bash
# 从 yolo11n 预训练权重开始训练（推荐起步）
python train.py --weights yolo11n.pt --epochs 150 --imgsz 640 --batch 16 --device 0

# 显存不足：减小 batch
python train.py --weights yolo11n.pt --epochs 150 --batch 8 --device 0

# 无 GPU
python train.py --weights yolo11n.pt --epochs 50 --batch 8 --device cpu

# 在已有权重上继续训练
python train.py --weights best.pt --epochs 50 --device 0

# 从上次中断处恢复
python train.py --weights yolo11n.pt --resume
```

常用参数（`python train.py --help` 查看全部）：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--weights` | `yolo11n.pt` | 预训练权重；`n/s/m/l/x` 精度与速度递增 |
| `--data` | `data/dnf.yaml` | 数据集配置 |
| `--epochs` | `150` | 训练轮数 |
| `--imgsz` | `640` | 输入尺寸 |
| `--batch` | `16` | 批大小 |
| `--device` | 自动 | `0`=GPU0，`cpu`=CPU |
| `--patience` | `50` | 早停耐心 |

训练产物在 `runs/detect/<name>/weights/best.pt`。

---

## 五、部署与使用

1. 把训练好的权重拷到项目根目录：

```bash
copy runs\detect\dnf\weights\best.pt best.pt
```

2. 运行主循环（自动挂机）：

```bash
python main2.py
```

3. 只查看检测效果（不操作游戏）：

```bash
python yolo11_detect.py
```

> 运行中按 `P` 暂停/继续，按 `q` 退出。
> 请先在活动小号上测试，注意账号安全（见 README 防封建议）。

---

## 六、迁移说明（重要）

- **旧 `best.pt` 是 YOLOv5 权重**，架构与 YOLOv11 不同。升级后需要用上面的流程重新训练得到 YOLOv11 的 `best.pt`。
  （ultralytics 也能加载旧 YOLOv5 权重做临时测试，但那仍是 v5 架构，不是 v11，也不推荐用于正式推理。）
- **`test/` 目录下的旧脚本**（`幽暗密林帕拉丁测试.py`、`未央.py`、`test/main2.py`）仍引用 `models.experimental` / `utils.general` 等旧 YOLOv5 代码。它们与主流程无关，属于历史测试；如需继续使用，可参照 `main2.py` 的改动自行迁移到 `yolo_detector.py`。
- `models/`、`utils/` 两个目录保留仅为兼容旧脚本，主流程不再依赖。

### 标注流程迁移：labelme → Label Studio（已完成）

| 项 | 旧（已废弃） | 新（当前） |
|----|-------------|-----------|
| 标注工具 | labelme | Label Studio（http://localhost:8080/projects/4 ） |
| 标注产物 | `xxx.jpg` + `xxx.json` 成对文件 | Label Studio 数据库内的标注记录 |
| 转换脚本 | `prepare_dataset.py` | `E:\Work\LabelStudio\export-yolo.ps1` |
| 类别顺序来源 | 脚本内硬编码 `NAME2ID` | 运行时读 `data/dnf.yaml` 的 `names` |

**`prepare_dataset.py` 与 `json2yolo.py`、`datasets_utils.py` 保留但已废弃**，
仅用于把历史上 labelme 标好的数据转成 YOLO 格式。运行时会打印废弃警告。

**不要再往仓库里新增 labelme 的 `xxx.json` 标注。** 所有新标注一律走 Label Studio。

旧数据转换（如仍有需要）：

```bash
python prepare_dataset.py --src path/to/labeled --dst datasets/dnf --val-ratio 0.1
```


