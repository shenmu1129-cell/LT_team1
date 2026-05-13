from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

from .utils import find_image_by_stem, load_json


@dataclass
class DetectionRecord:
    image_path: Path
    boxes: list[list[float]]
    labels: list[str]


def _normalize_label(label: str, aliases: dict[str, str] | None) -> str:
    label = str(label).strip()
    if aliases and label in aliases:
        return str(aliases[label]).strip()
    return label


def _valid_box(box: Iterable[float]) -> bool:
    x1, y1, x2, y2 = [float(v) for v in box]
    return x2 > x1 and y2 > y1


def _bbox_from_tt100k(obj: dict) -> list[float] | None:
    bbox = obj.get("bbox") or obj.get("box") or obj.get("rect")
    if bbox is None:
        return None
    if isinstance(bbox, dict):
        keys = bbox.keys()
        if {"xmin", "ymin", "xmax", "ymax"}.issubset(keys):
            return [bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]]
        if {"x1", "y1", "x2", "y2"}.issubset(keys):
            return [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]]
        if {"left", "top", "right", "bottom"}.issubset(keys):
            return [bbox["left"], bbox["top"], bbox["right"], bbox["bottom"]]
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        return [float(v) for v in bbox]
    return None


def _split_name_from_path(path: str) -> str:
    parts = Path(path).parts
    for split in ("train", "test", "other", "val", "valid", "validation"):
        if split in parts:
            return "val" if split in {"valid", "validation"} else split
    return ""


def load_tt100k_records(cfg: dict, split_names: list[str]) -> list[DetectionRecord]:
    root = Path(cfg["root"]).expanduser()
    annotation_file = root / cfg.get("annotation_file", "data/annotations.json")
    image_root = root / cfg.get("image_root", "data")
    aliases = cfg.get("class_aliases") or {}

    data = load_json(annotation_file)
    imgs = data.get("imgs", data)
    records: list[DetectionRecord] = []
    wanted = set(split_names)

    for image_id, image_data in imgs.items():
        rel_path = image_data.get("path") or image_data.get("file_name") or image_id
        split_name = image_data.get("split") or _split_name_from_path(rel_path)
        if wanted and split_name not in wanted:
            continue

        image_path = image_root / rel_path
        if not image_path.exists() and image_path.name != Path(rel_path).name:
            image_path = image_root / Path(rel_path).name
        if not image_path.exists():
            image_path = root / rel_path

        boxes: list[list[float]] = []
        labels: list[str] = []
        objects = image_data.get("objects") or image_data.get("marks") or []
        for obj in objects:
            label = obj.get("category") or obj.get("label") or obj.get("name")
            box = _bbox_from_tt100k(obj)
            if not label or box is None or not _valid_box(box):
                continue
            boxes.append([float(v) for v in box])
            labels.append(_normalize_label(label, aliases))

        if boxes:
            records.append(DetectionRecord(image_path=image_path, boxes=boxes, labels=labels))

    return records


def _resolve_path(root: Path, path_text: str) -> Path:
    path = Path(path_text.strip())
    if path.is_absolute():
        return path
    return root / path


def _yolo_label_path(root: Path, image_path: Path) -> Path:
    try:
        rel = image_path.relative_to(root)
    except ValueError:
        rel = image_path.name
    rel_path = Path(rel)
    parts = list(rel_path.parts)
    if "images" in parts:
        parts[parts.index("images")] = "labels"
        return root.joinpath(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def _yolo_to_xyxy(values: list[float], width: int, height: int) -> list[float]:
    x_center, y_center, box_width, box_height = values
    x1 = (x_center - box_width / 2.0) * width
    y1 = (y_center - box_height / 2.0) * height
    x2 = (x_center + box_width / 2.0) * width
    y2 = (y_center + box_height / 2.0) * height
    return [
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    ]


def load_yolo_records(cfg: dict, split: str) -> list[DetectionRecord]:
    root = Path(cfg["root"]).expanduser()
    split_file = root / cfg.get(f"{split}_file", f"{split}.txt")
    aliases = {str(index): str(name) for index, name in enumerate(cfg.get("classes") or [])}
    aliases.update(cfg.get("class_aliases") or {})
    skip_empty = bool(cfg.get("skip_empty", True))

    records: list[DetectionRecord] = []
    with split_file.open("r", encoding="utf-8") as f:
        image_paths = [_resolve_path(root, line.split()[0]) for line in f if line.strip()]

    for image_path in image_paths:
        if not image_path.exists():
            continue

        label_path = _yolo_label_path(root, image_path)
        boxes: list[list[float]] = []
        labels: list[str] = []
        if label_path.exists():
            with Image.open(image_path) as image:
                width, height = image.size
            with label_path.open("r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    label = _normalize_label(parts[0], aliases)
                    try:
                        box = _yolo_to_xyxy([float(v) for v in parts[1:5]], width, height)
                    except ValueError:
                        continue
                    if not _valid_box(box):
                        continue
                    boxes.append(box)
                    labels.append(label)

        if boxes or not skip_empty:
            records.append(DetectionRecord(image_path=image_path, boxes=boxes, labels=labels))

    return records


def _read_split_file(path: Path | None) -> set[str] | None:
    if path is None or not path.exists():
        return None
    names: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                names.add(Path(line.split()[0]).stem)
    return names


def load_cctsdb_records(cfg: dict, split: str) -> list[DetectionRecord]:
    root = Path(cfg["root"]).expanduser()
    image_dir = root / cfg.get("image_dir", "images")
    annotation_dir = root / cfg.get("annotation_dir", "annotations")
    aliases = cfg.get("class_aliases") or {}
    seed = int(cfg.get("seed", 42))
    val_ratio = float(cfg.get("val_ratio", 0.2))

    train_names = _read_split_file(root / cfg["train_split_file"]) if cfg.get("train_split_file") else None
    val_names = _read_split_file(root / cfg["val_split_file"]) if cfg.get("val_split_file") else None

    xml_paths = sorted(annotation_dir.rglob("*.xml"))
    if train_names is None and val_names is None:
        stems = [p.stem for p in xml_paths]
        rng = random.Random(seed)
        rng.shuffle(stems)
        val_count = max(1, int(len(stems) * val_ratio))
        val_names = set(stems[:val_count])
        train_names = set(stems[val_count:])

    selected = train_names if split == "train" else val_names
    records: list[DetectionRecord] = []

    for xml_path in xml_paths:
        if selected is not None and xml_path.stem not in selected:
            continue
        tree = ET.parse(xml_path)
        root_xml = tree.getroot()
        filename = root_xml.findtext("filename") or f"{xml_path.stem}.jpg"
        image_path = image_dir / filename
        if not image_path.exists():
            found = find_image_by_stem(image_dir, xml_path.stem)
            if found is not None:
                image_path = found

        boxes: list[list[float]] = []
        labels: list[str] = []
        for obj in root_xml.findall("object"):
            label = obj.findtext("name")
            bndbox = obj.find("bndbox")
            if not label or bndbox is None:
                continue
            box = [
                float(bndbox.findtext("xmin", "0")),
                float(bndbox.findtext("ymin", "0")),
                float(bndbox.findtext("xmax", "0")),
                float(bndbox.findtext("ymax", "0")),
            ]
            if not _valid_box(box):
                continue
            boxes.append(box)
            labels.append(_normalize_label(label, aliases))

        if boxes:
            records.append(DetectionRecord(image_path=image_path, boxes=boxes, labels=labels))

    return records


class TrafficSignDetectionDataset(Dataset):
    def __init__(
        self,
        records: list[DetectionRecord],
        class_to_idx: dict[str, int],
        train: bool = False,
    ) -> None:
        self.records = records
        self.class_to_idx = class_to_idx
        self.train = train

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = Image.open(record.image_path).convert("RGB")
        boxes = torch.as_tensor(record.boxes, dtype=torch.float32)
        labels = torch.as_tensor([self.class_to_idx[name] for name in record.labels], dtype=torch.int64)
        image_id = torch.tensor([index], dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        image_tensor = F.to_tensor(image)
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
        }
        return image_tensor, target


def build_records(cfg: dict, split: str) -> list[DetectionRecord]:
    dataset_cfg = dict(cfg["dataset"])
    dataset_cfg["seed"] = cfg.get("train", {}).get("seed", 42)
    name = dataset_cfg["name"].lower()
    if name == "tt100k":
        split_key = f"{split}_splits"
        return load_tt100k_records(dataset_cfg, dataset_cfg.get(split_key, [split]))
    if name in {"tt100k_yolo", "yolo"}:
        return load_yolo_records(dataset_cfg, split)
    if name == "cctsdb":
        return load_cctsdb_records(dataset_cfg, split)
    raise ValueError(f"Unsupported dataset: {dataset_cfg['name']}")


def build_class_mapping(train_records: list[DetectionRecord], configured_classes: list[str] | None):
    if configured_classes:
        classes = [str(name) for name in configured_classes]
    else:
        classes = sorted({label for record in train_records for label in record.labels})
        if classes and all(label.isdigit() for label in classes):
            classes = sorted(classes, key=lambda label: int(label))
    return {name: idx + 1 for idx, name in enumerate(classes)}, classes


def filter_records_by_classes(records: list[DetectionRecord], class_to_idx: dict[str, int]) -> list[DetectionRecord]:
    filtered: list[DetectionRecord] = []
    for record in records:
        boxes = []
        labels = []
        for box, label in zip(record.boxes, record.labels):
            if label in class_to_idx:
                boxes.append(box)
                labels.append(label)
        if boxes:
            filtered.append(DetectionRecord(record.image_path, boxes, labels))
    return filtered
