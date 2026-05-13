from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as F

from src.model import build_faster_rcnn
from src.utils import load_json


def parse_args():
    parser = argparse.ArgumentParser(description="Run Faster R-CNN inference on one image.")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pth or last.pth.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--classes", required=True, help="Path to classes.json saved by train.py.")
    parser.add_argument("--output", default="outputs/prediction.jpg", help="Output image path.")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    meta = load_json(args.classes)
    classes = meta["classes"]

    checkpoint = torch.load(args.checkpoint, map_location=device)
    cfg = checkpoint["config"]
    model = build_faster_rcnn(num_classes=len(classes) + 1, cfg=cfg)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    image = Image.open(args.image).convert("RGB")
    image_tensor = F.to_tensor(image).to(device)
    with torch.no_grad():
        pred = model([image_tensor])[0]

    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font = None

    for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
        score_value = float(score.item())
        if score_value < args.score_threshold:
            continue
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        class_name = classes[int(label.item()) - 1]
        text = f"{class_name} {score_value:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle(text_bbox, fill="red")
        draw.text((x1, y1), text, fill="white", font=font)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    print(f"Saved prediction to: {output}")


if __name__ == "__main__":
    main()
