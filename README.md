# Faster R-CNN for TT100K / CCTSDB

这是一套基于 `torchvision` 封装版 Faster R-CNN 的交通标志目标检测训练代码，支持：

- TT100K YOLO txt 标注，也就是你现在的 `TT100K-2016/train.txt + train/labels/*.txt` 结构
- TT100K 原始 `annotations.json`
- CCTSDB / CCTSDB 2021 常见 VOC XML 标注

## 需要准备的数据文件

### TT100K

你服务器上的 TT100K 是 YOLO 格式，目录类似：

```text
TT100K-2016/
  train.txt
  test.txt
  train/
    images/
      *.jpg
    labels/
      *.txt
  test/
    images/
      *.jpg
    labels/
      *.txt
```

你至少需要：

- `train.txt`
- `test.txt`
- `train/images/` 和 `train/labels/`
- `test/images/` 和 `test/labels/`

每个 label txt 的格式应为 YOLO 检测格式：

```text
class_id x_center y_center width height
```

其中 `x_center/y_center/width/height` 是 0 到 1 的归一化值。代码会自动转换成 Faster R-CNN 需要的像素坐标 `[xmin, ymin, xmax, ymax]`。

### CCTSDB

如果你的 CCTSDB 是目录式 YOLO 格式，目录类似：

```text
CCTSDB/
  images/
    train/
      *.jpg
    test/
      *.jpg
    test_fgsm_v9/
      *.jpg
  labels/
    train/
      *.txt
    test/
      *.txt
    test_fgsm_v9/
      *.txt
```

你至少需要：

- `images/train` 和 `labels/train`
- `images/test` 和 `labels/test`

代码会自动把 YOLO 归一化框转换成 Faster R-CNN 需要的像素框。`images/test_fgsm_v9` 可以作为对抗样本目录，用 `evaluate.py --adv-root` 评估。

## 安装

Ubuntu 上建议先按你的 CUDA 版本安装 PyTorch，然后再装其余依赖：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

如果只用 CPU：

```bash
pip install torch torchvision
pip install -r requirements.txt
```

## 训练 TT100K

`configs/tt100k.yaml` 已经内置 TT100K YOLO 格式和 45 个类别名。可以直接输入数据集路径训练：

```bash
python train.py --data-root /home/sutongtong/LanTu_team1/TT100K-2016
```

如果想指定输出目录：

```bash
python train.py \
  --data-root /home/sutongtong/LanTu_team1/TT100K-2016 \
  --output-dir outputs/tt100k_fasterrcnn
```

如果显存不够：

```bash
python train.py \
  --data-root /home/sutongtong/LanTu_team1/TT100K-2016 \
  --batch-size 1
```

### 后台训练脚本

可以直接用脚本在指定显卡后台训练：

```bash
GPU_ID=2 bash scripts/train_tt100k_bg.sh
```

默认配置：

```text
epochs=50
batch_size=4
lr=0.003
trainable_backbone_layers=5
min_size=1024
max_size=1600
eval_map_every=10
quick_eval_samples=100
output_dir=outputs/tt100k_frcnn_ep50_bs4_lr003
```

查看日志：

```bash
tail -f logs/tt100k_frcnn_ep50_bs4_lr003.log
```

停止训练：

```bash
kill $(cat outputs/tt100k_frcnn_ep50_bs4_lr003/train.pid)
```

如果想改数据集路径、实验名或显卡号：

```bash
GPU_ID=3 \
DATA_ROOT=/home/sutongtong/LanTu_team1/TT100K-2016 \
EXP_NAME=tt100k_frcnn_ep50_gpu3 \
bash scripts/train_tt100k_bg.sh
```

底层配置等价于：

```yaml
dataset:
  name: tt100k_yolo
  root: /home/sutongtong/LanTu_team1/TT100K-2016
  train_file: train.txt
  val_file: test.txt
```

当然，也可以显式指定配置文件：

```bash
python train.py --config configs/tt100k.yaml
```

训练前可以先检查一下数据：

```bash
python tools/check_yolo_dataset.py --root /home/sutongtong/LanTu_team1/TT100K-2016 --split-file train.txt
python tools/check_yolo_dataset.py --root /home/sutongtong/LanTu_team1/TT100K-2016 --split-file test.txt
```

## 训练 CCTSDB

先改 `configs/cctsdb.yaml`：

```yaml
dataset:
  name: cctsdb_yolo
  root: /absolute/path/to/CCTSDB
  image_dir: images
  label_dir: labels
  train_split: train
  val_split: test
```

然后运行：

```bash
python train.py --config configs/cctsdb.yaml
```

也可以直接用后台脚本：

```bash
GPU_ID=2 bash scripts/train_cctsdb_bg.sh
```

默认配置：

```text
data_root=/home/sutongtong/LanTu_team1/advYOLO+AdaAD+CCTSDB/CCTSDB2021
epochs=80
batch_size=8
lr=0.005
trainable_backbone_layers=5
min_size=1024
max_size=1600
eval_map_every=10
quick_eval_samples=100
output_dir=outputs/cctsdb_frcnn_ep80_bs8_lr005
```

查看日志：

```bash
tail -f logs/cctsdb_frcnn_ep80_bs8_lr005.log
```

如果 CCTSDB 的类别名还不确定，配置里的 `classes: null` 会自动使用标签中的数字类别，例如 `0`、`1`、`2`。如果你有类别名，建议改成：

```yaml
classes:
  - mandatory
  - prohibitory
  - warning
```

实际顺序必须和你的 YOLO 标签 id 一致。

## 指定类别

默认会从训练集标注中自动收集类别。如果你只想训练某些类别，在配置中写：

```yaml
classes:
  - mandatory
  - prohibitory
  - warning
```

注意：Faster R-CNN 内部的背景类不需要写，代码会自动加上。

## 预测单张图片

```bash
python predict.py \
  --checkpoint outputs/cctsdb_fasterrcnn/best.pth \
  --classes outputs/cctsdb_fasterrcnn/classes.json \
  --image /path/to/test.jpg \
  --output outputs/prediction.jpg \
  --score-threshold 0.5
```

## 评估 mAP50 / Recall / ASR

训练完成后，可以先在干净测试集上评估：

```bash
python evaluate.py \
  --checkpoint outputs/tt100k_fasterrcnn_bs4/best.pth \
  --data-root /home/sutongtong/LanTu_team1/TT100K-2016 \
  --source-model YOLOv10 \
  --target-detector "Faster R-CNN" \
  --output-csv outputs/metrics.csv
```

如果只想先抽一部分样本快速测试，例如 200 张：

```bash
python evaluate.py \
  --checkpoint outputs/tt100k_fasterrcnn_bs4/best.pth \
  --data-root /home/sutongtong/LanTu_team1/TT100K-2016 \
  --max-samples 200 \
  --output-csv outputs/metrics_quick.csv
```

如果你有对抗样本文件夹，并且对抗图片文件名和 TT100K 测试集图片文件名一致，可以同时评估干净集和对抗集：

```bash
python evaluate.py \
  --checkpoint outputs/tt100k_fasterrcnn_bs4/best.pth \
  --data-root /home/sutongtong/LanTu_team1/TT100K-2016 \
  --adv-root "/home/sutongtong/advYOLO+AdaADv4/TOGyolov9对抗样本" \
  --source-model YOLOv10 \
  --target-detector "Faster R-CNN" \
  --batch-size 8 \
  --output-csv outputs/metrics.csv
```

输出会包含：

```text
Source Model, Target Detector, Clean mAP50, Adv mAP50, Clean Recall, Adv Recall, ASR
```

其中 ASR 的计算方式是：干净图中已经成功检出的目标，在对抗图中未被成功检出的比例。

## 说明

- `train.py` 保存 `last.pth` 和验证损失最小的 `best.pth`。
- 后台训练脚本会每 10 轮输出一次验证集 `val_mAP50`，并保存 mAP50 最高的 `best_map50.pth`。
- 后台训练脚本会每轮用验证集前 100 张做快速评估，输出 `quick_mAP50` 和 `quick_recall`，并追加到 `quick_eval.csv`。
- `classes.json` 保存类别顺序，预测时必须和 checkpoint 一起使用。
- 显存不够时，优先把 `batch_size` 改为 `1`。
- 你当前 TT100K 标注里类别编号是 `0` 到 `44`，配置中已经固定为 45 类类别名。Faster R-CNN 训练时会自动额外加入背景类。
- YOLO 标签里的数字 id 会按 `classes` 顺序自动映射，比如 `26 -> pl40`。
