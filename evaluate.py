from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as F
from tqdm import tqdm

from src.datasets import DetectionRecord, build_class_mapping, build_records, filter_records_by_classes
from src.model import build_faster_rcnn
from src.utils import IMAGE_EXTENSIONS, load_json, load_yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Faster R-CNN and export mAP50/Recall/ASR table.")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pth or last.pth.")
    parser.add_argument("--classes", default=None, help="Path to classes.json saved by train.py.")
    parser.add_argument("--config", default="configs/tt100k.yaml", help="Dataset config.")
    parser.add_argument("--data-root", default=None, help="Clean dataset root.")
    parser.add_argument("--split", default="val", choices=["train", "val"], help="Clean split to evaluate.")
    parser.add_argument("--adv-root", default=None, help="Adversarial image folder. Uses clean GT labels.")
    parser.add_argument("--adv-suffix", default="", help="Suffix added to clean image stems, e.g. _adv.")
    parser.add_argument("--adv-prefix", default="", help="Prefix added to clean image stems.")
    parser.add_argument("--source-model", default="YOLOv10", help="Source model name for table.")
    parser.add_argument("--target-detector", default="Faster R-CNN", help="Target detector name for table.")
    parser.add_argument("--score-threshold", type=float, default=0.05, help="Score threshold for recall/ASR.")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for mAP50/Recall/ASR.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument("--max-samples", type=int, default=None, help="Evaluate only the first N clean samples.")
    parser.add_argument("--output-csv", default="outputs/eval_metrics.csv")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)


def ap_from_pr(tp: list[int], fp: list[int], total_gt: int) -> float:
    if total_gt == 0:
        return float("nan")
    tp_cum = torch.tensor(tp, dtype=torch.float32).cumsum(0)
    fp_cum = torch.tensor(fp, dtype=torch.float32).cumsum(0)
    recalls = tp_cum / max(1, total_gt)
    precisions = tp_cum / torch.clamp(tp_cum + fp_cum, min=1e-6)

    recalls = torch.cat([torch.tensor([0.0]), recalls, torch.tensor([1.0])])
    precisions = torch.cat([torch.tensor([0.0]), precisions, torch.tensor([0.0])])
    for index in range(precisions.numel() - 2, -1, -1):
        precisions[index] = torch.maximum(precisions[index], precisions[index + 1])
    changed = torch.where(recalls[1:] != recalls[:-1])[0]
    return float(torch.sum((recalls[changed + 1] - recalls[changed]) * precisions[changed + 1]).item())


def evaluate_predictions(
    predictions: list[dict[str, torch.Tensor]],
    records: list[DetectionRecord],
    class_to_idx: dict[str, int],
    score_threshold: float,
    iou_threshold: float,
):
    idx_to_class = {index: name for name, index in class_to_idx.items()}
    gt_by_class: dict[int, dict[int, torch.Tensor]] = defaultdict(dict)
    total_gt_by_class: dict[int, int] = defaultdict(int)

    for image_index, record in enumerate(records):
        boxes = torch.tensor(record.boxes, dtype=torch.float32)
        labels = torch.tensor([class_to_idx[label] for label in record.labels], dtype=torch.int64)
        for class_idx in idx_to_class:
            class_boxes = boxes[labels == class_idx]
            gt_by_class[class_idx][image_index] = class_boxes
            total_gt_by_class[class_idx] += int(class_boxes.shape[0])

    ap_values: list[float] = []
    for class_idx in idx_to_class:
        pred_items: list[tuple[float, int, torch.Tensor]] = []
        for image_index, pred in enumerate(predictions):
            keep = pred["labels"] == class_idx
            for score, box in zip(pred["scores"][keep], pred["boxes"][keep]):
                pred_items.append((float(score.item()), image_index, box.cpu()))
        pred_items.sort(key=lambda item: item[0], reverse=True)

        matched: dict[int, set[int]] = defaultdict(set)
        tp: list[int] = []
        fp: list[int] = []
        for _, image_index, pred_box in pred_items:
            gt_boxes = gt_by_class[class_idx].get(image_index, torch.empty((0, 4)))
            if gt_boxes.numel() == 0:
                tp.append(0)
                fp.append(1)
                continue
            ious = box_iou(pred_box.unsqueeze(0), gt_boxes).squeeze(0)
            best_iou, best_gt = torch.max(ious, dim=0)
            gt_index = int(best_gt.item())
            if float(best_iou.item()) >= iou_threshold and gt_index not in matched[image_index]:
                matched[image_index].add(gt_index)
                tp.append(1)
                fp.append(0)
            else:
                tp.append(0)
                fp.append(1)

        ap = ap_from_pr(tp, fp, total_gt_by_class[class_idx])
        if not torch.isnan(torch.tensor(ap)):
            ap_values.append(ap)

    detected_gt: set[tuple[int, int]] = set()
    total_gt = 0
    for image_index, (record, pred) in enumerate(zip(records, predictions)):
        gt_boxes = torch.tensor(record.boxes, dtype=torch.float32)
        gt_labels = torch.tensor([class_to_idx[label] for label in record.labels], dtype=torch.int64)
        total_gt += int(gt_boxes.shape[0])
        pred_keep = pred["scores"] >= score_threshold
        pred_boxes = pred["boxes"][pred_keep].cpu()
        pred_labels = pred["labels"][pred_keep].cpu()

        for gt_index, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
            same_class = pred_labels == gt_label
            if same_class.sum() == 0:
                continue
            ious = box_iou(gt_box.unsqueeze(0), pred_boxes[same_class]).squeeze(0)
            if ious.numel() and float(ious.max().item()) >= iou_threshold:
                detected_gt.add((image_index, gt_index))

    mean_ap50 = sum(ap_values) / max(1, len(ap_values))
    recall = len(detected_gt) / max(1, total_gt)
    return {
        "map50": mean_ap50 * 100.0,
        "recall": recall * 100.0,
        "detected_gt": detected_gt,
        "total_gt": total_gt,
    }


def load_classes(checkpoint: dict, classes_path: str | None) -> list[str]:
    if classes_path:
        return load_json(classes_path)["classes"]
    if "classes" in checkpoint:
        return [str(name) for name in checkpoint["classes"]]
    raise RuntimeError("No class metadata found. Pass --classes outputs/.../classes.json.")


def load_model(checkpoint_path: str, classes: list[str], device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint["config"]
    model = build_faster_rcnn(num_classes=len(classes) + 1, cfg=cfg)
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint


def find_adv_images(adv_root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for path in adv_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            paths.setdefault(path.name, path)
            paths.setdefault(path.stem, path)
    return paths


def records_with_adv_images(
    clean_records: list[DetectionRecord], adv_root: str, adv_prefix: str = "", adv_suffix: str = ""
) -> tuple[list[int], list[DetectionRecord], list[DetectionRecord], int]:
    adv_paths = find_adv_images(Path(adv_root).expanduser())
    matched_indices: list[int] = []
    matched_clean_records: list[DetectionRecord] = []
    records: list[DetectionRecord] = []
    missing = 0
    for index, record in enumerate(clean_records):
        clean_stem = record.image_path.stem
        clean_name = record.image_path.name
        adv_stem = f"{adv_prefix}{clean_stem}{adv_suffix}"
        candidates = [
            clean_name,
            clean_stem,
            adv_stem,
        ]
        adv_path = None
        for key in candidates:
            adv_path = adv_paths.get(key)
            if adv_path is not None:
                break
        if adv_path is None:
            missing += 1
            continue
        matched_indices.append(index)
        matched_clean_records.append(record)
        records.append(DetectionRecord(adv_path, record.boxes, record.labels))
    return matched_indices, matched_clean_records, records, missing


def run_inference(model, records: list[DetectionRecord], device: torch.device, batch_size: int, max_detections: int):
    predictions: list[dict[str, torch.Tensor]] = []
    for start in tqdm(range(0, len(records), batch_size), desc="Infer"):
        batch_records = records[start : start + batch_size]
        images = []
        for record in batch_records:
            image = Image.open(record.image_path).convert("RGB")
            images.append(F.to_tensor(image).to(device))
        with torch.no_grad():
            outputs = model(images)
        for output in outputs:
            predictions.append(
                {
                    "boxes": output["boxes"][:max_detections].detach().cpu(),
                    "labels": output["labels"][:max_detections].detach().cpu(),
                    "scores": output["scores"][:max_detections].detach().cpu(),
                }
            )
    return predictions


def format_value(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.1f}"


def print_markdown_table(row: dict[str, str]) -> None:
    headers = ["Source Model", "Target Detector", "Clean mAP50", "Adv mAP50", "Clean Recall", "Adv Recall", "ASR"]
    print()
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    print("| " + " | ".join(row[header] for header in headers) + " |")


def save_csv(row: dict[str, str], output_csv: str) -> None:
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"Saved metrics to: {path}")


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(args.checkpoint, map_location=device)
    classes = load_classes(checkpoint, args.classes)
    model, _ = load_model(args.checkpoint, classes, device)

    cfg = load_yaml(args.config)
    if args.data_root:
        cfg["dataset"]["root"] = args.data_root
    class_to_idx = {name: index + 1 for index, name in enumerate(classes)}

    records = build_records(cfg, args.split)
    records = filter_records_by_classes(records, class_to_idx)
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be a positive integer.")
        records = records[: args.max_samples]
    if not records:
        raise RuntimeError("No evaluation samples found. Check --data-root and --split.")

    print(f"Device: {device}")
    print(f"Classes: {len(classes)} | Clean samples: {len(records)}")
    clean_predictions = run_inference(model, records, device, args.batch_size, args.max_detections)
    clean_metrics = evaluate_predictions(
        clean_predictions, records, class_to_idx, args.score_threshold, args.iou_threshold
    )
    print(
        f"Clean mAP50={clean_metrics['map50']:.2f}, "
        f"Clean Recall={clean_metrics['recall']:.2f}, GT boxes={clean_metrics['total_gt']}"
    )

    adv_metrics = None
    attack_success = None
    if args.adv_root:
        matched_indices, matched_clean_records, adv_records, missing = records_with_adv_images(
            records,
            args.adv_root,
            adv_prefix=args.adv_prefix,
            adv_suffix=args.adv_suffix,
        )
        if not adv_records:
            raise RuntimeError("No adversarial images matched clean validation filenames.")
        if missing:
            print(f"Warning: {missing} clean images were not found in adversarial folder.")
        print(f"Adv samples: {len(adv_records)} matched from {len(records)} clean samples")
        adv_predictions = run_inference(model, adv_records, device, args.batch_size, args.max_detections)
        adv_metrics = evaluate_predictions(
            adv_predictions, adv_records, class_to_idx, args.score_threshold, args.iou_threshold
        )
        matched_clean_predictions = [clean_predictions[index] for index in matched_indices]
        matched_clean_metrics = evaluate_predictions(
            matched_clean_predictions,
            matched_clean_records,
            class_to_idx,
            args.score_threshold,
            args.iou_threshold,
        )
        clean_detected = matched_clean_metrics["detected_gt"]
        adv_detected = adv_metrics["detected_gt"]
        attack_success = len(clean_detected - adv_detected) / max(1, len(clean_detected)) * 100.0
        print(
            f"Adv mAP50={adv_metrics['map50']:.2f}, "
            f"Adv Recall={adv_metrics['recall']:.2f}, ASR={attack_success:.2f}"
        )
    else:
        print("No --adv-root provided. Adv mAP50, Adv Recall, and ASR will be written as NA.")

    adv_map50 = adv_metrics["map50"] if adv_metrics else None
    adv_recall = adv_metrics["recall"] if adv_metrics else None

    row = {
        "Source Model": args.source_model,
        "Target Detector": args.target_detector,
        "Clean mAP50": format_value(clean_metrics["map50"]),
        "Adv mAP50": format_value(adv_map50),
        "Clean Recall": format_value(clean_metrics["recall"]),
        "Adv Recall": format_value(adv_recall),
        "ASR": format_value(attack_success),
    }
    print_markdown_table(row)
    save_csv(row, args.output_csv)


if __name__ == "__main__":
    main()
