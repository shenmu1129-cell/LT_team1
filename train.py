from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import (
    TrafficSignDetectionDataset,
    build_class_mapping,
    build_records,
    filter_records_by_classes,
)
from src.model import build_faster_rcnn
from src.utils import collate_fn, load_yaml, move_targets_to_device, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train TorchVision Faster R-CNN on TT100K or CCTSDB.")
    parser.add_argument("--config", default="configs/tt100k.yaml", help="Path to YAML config.")
    parser.add_argument("--data-root", default=None, help="Override dataset root path.")
    parser.add_argument("--output-dir", default=None, help="Override checkpoint output directory.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override dataloader workers.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--lr-step-size", type=int, default=None, help="Override StepLR step size.")
    parser.add_argument("--lr-gamma", type=float, default=None, help="Override StepLR gamma.")
    parser.add_argument("--weight-decay", type=float, default=None, help="Override weight decay.")
    parser.add_argument("--trainable-backbone-layers", type=int, default=None, help="Fine-tuned backbone layers.")
    parser.add_argument("--min-size", type=int, default=None, help="Detector input min size.")
    parser.add_argument("--max-size", type=int, default=None, help="Detector input max size.")
    parser.add_argument("--device", default=None, help="cuda, cpu, or leave empty for auto.")
    return parser.parse_args()


def apply_cli_overrides(cfg: dict, args) -> dict:
    if args.data_root:
        cfg["dataset"]["root"] = args.data_root
    if args.output_dir:
        cfg["train"]["output_dir"] = args.output_dir
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["train"]["num_workers"] = args.num_workers
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.lr_step_size is not None:
        cfg["train"]["lr_step_size"] = args.lr_step_size
    if args.lr_gamma is not None:
        cfg["train"]["lr_gamma"] = args.lr_gamma
    if args.weight_decay is not None:
        cfg["train"]["weight_decay"] = args.weight_decay
    if args.trainable_backbone_layers is not None:
        cfg["model"]["trainable_backbone_layers"] = args.trainable_backbone_layers
    if args.min_size is not None:
        cfg["model"]["min_size"] = args.min_size
    if args.max_size is not None:
        cfg["model"]["max_size"] = args.max_size
    return cfg


def evaluate_loss(model, loader, device):
    model.train()
    total = 0.0
    batches = 0
    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            targets = move_targets_to_device(targets, device)
            losses = model(images, targets)
            loss = sum(value for value in losses.values())
            total += float(loss.item())
            batches += 1
    return total / max(1, batches)


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    cfg = apply_cli_overrides(cfg, args)
    train_cfg = cfg["train"]
    set_seed(int(train_cfg.get("seed", 42)))

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_records = build_records(cfg, "train")
    val_records = build_records(cfg, "val")
    class_to_idx, classes = build_class_mapping(train_records, cfg["dataset"].get("classes"))
    train_records = filter_records_by_classes(train_records, class_to_idx)
    val_records = filter_records_by_classes(val_records, class_to_idx)

    if not train_records:
        raise RuntimeError("No training samples found. Please check dataset paths and split settings.")
    if not val_records:
        print("Warning: no validation samples found. Training will continue without validation loss.")

    save_json({"classes": classes, "class_to_idx": class_to_idx}, output_dir / "classes.json")
    print(f"Device: {device}")
    print(f"Classes: {len(classes)}")
    print(f"Train samples: {len(train_records)} | Val samples: {len(val_records)}")

    train_dataset = TrafficSignDetectionDataset(train_records, class_to_idx, train=True)
    val_dataset = TrafficSignDetectionDataset(val_records, class_to_idx, train=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg.get("batch_size", 2)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 4)),
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 4)),
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    model = build_faster_rcnn(num_classes=len(classes) + 1, cfg=cfg).to(device)
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(train_cfg.get("lr", 0.005)),
        momentum=float(train_cfg.get("momentum", 0.9)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0005)),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(train_cfg.get("lr_step_size", 8)),
        gamma=float(train_cfg.get("lr_gamma", 0.1)),
    )

    start_epoch = 0
    resume = train_cfg.get("resume")
    if resume:
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1

    best_val = float("inf")
    epochs = int(train_cfg.get("epochs", 20))
    print_freq = int(train_cfg.get("print_freq", 50))
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        start = time.time()
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for step, (images, targets) in enumerate(progress, start=1):
            images = [img.to(device) for img in images]
            targets = move_targets_to_device(targets, device)
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad(set_to_none=True)
            losses.backward()
            optimizer.step()

            loss_value = float(losses.item())
            epoch_loss += loss_value
            if step % print_freq == 0 or step == 1:
                progress.set_postfix(loss=f"{loss_value:.4f}", lr=optimizer.param_groups[0]["lr"])

        scheduler.step()
        train_loss = epoch_loss / max(1, len(train_loader))
        val_loss = evaluate_loss(model, val_loader, device) if val_records else float("nan")
        elapsed = time.time() - start
        print(
            f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, time={elapsed:.1f}s"
        )

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "classes": classes,
            "config": cfg,
        }
        torch.save(checkpoint, output_dir / "last.pth")
        if val_records and val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, output_dir / "best.pth")

    print(f"Done. Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()
