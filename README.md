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

建议目录：

```text
CCTSDB/
  images/
    *.jpg
  annotations/
    *.xml
```

你至少需要：

- 图片目录，如 `images/`
- VOC XML 标注目录，如 `annotations/`

如果你有官方划分文件，也可以在 `configs/cctsdb.yaml` 里填：

```yaml
train_split_file: ImageSets/Main/train.txt
val_split_file: ImageSets/Main/val.txt
```

没有划分文件时，代码会自动按 `val_ratio` 从 XML 中划分训练集和验证集。

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
  root: /absolute/path/to/CCTSDB
  image_dir: images
  annotation_dir: annotations
```

然后运行：

```bash
python train.py --config configs/cctsdb.yaml
```

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

## 说明

- `train.py` 保存 `last.pth` 和验证损失最小的 `best.pth`。
- `classes.json` 保存类别顺序，预测时必须和 checkpoint 一起使用。
- 显存不够时，优先把 `batch_size` 改为 `1`。
- 你当前 TT100K 标注里类别编号是 `0` 到 `44`，配置中已经固定为 45 类类别名。Faster R-CNN 训练时会自动额外加入背景类。
- YOLO 标签里的数字 id 会按 `classes` 顺序自动映射，比如 `26 -> pl40`。
