from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect YOLO-format detection dataset.")
    parser.add_argument("--root", required=True, help="Dataset root, for example /home/.../TT100K-2016")
    parser.add_argument("--split-file", default="train.txt", help="Image list file under root.")
    return parser.parse_args()


def label_path_for(root: Path, image_path: Path) -> Path:
    rel = image_path.relative_to(root) if image_path.is_relative_to(root) else Path(image_path.name)
    parts = list(rel.parts)
    if "images" in parts:
        parts[parts.index("images")] = "labels"
        return root.joinpath(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def main():
    args = parse_args()
    root = Path(args.root).expanduser()
    split_file = root / args.split_file
    image_paths = []
    with split_file.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                path = Path(line.split()[0])
                image_paths.append(path if path.is_absolute() else root / path)

    missing_images = 0
    missing_labels = 0
    empty_labels = 0
    class_counts: Counter[str] = Counter()
    bad_lines = 0
    for image_path in image_paths:
        if not image_path.exists() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            missing_images += 1
            continue
        label_path = label_path_for(root, image_path)
        if not label_path.exists():
            missing_labels += 1
            continue
        lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            empty_labels += 1
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                bad_lines += 1
                continue
            class_counts[parts[0]] += 1

    print(f"split_file: {split_file}")
    print(f"images_in_split: {len(image_paths)}")
    print(f"missing_images: {missing_images}")
    print(f"missing_labels: {missing_labels}")
    print(f"empty_labels: {empty_labels}")
    print(f"bad_label_lines: {bad_lines}")
    print("classes:", " ".join(sorted(class_counts, key=lambda x: int(x) if x.isdigit() else x)))
    print("class_counts:")
    for cls in sorted(class_counts, key=lambda x: int(x) if x.isdigit() else x):
        print(f"  {cls}: {class_counts[cls]}")


if __name__ == "__main__":
    main()
