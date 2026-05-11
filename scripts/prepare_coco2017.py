#!/usr/bin/env python3
"""Prepare COCO 2017 captions for the existing Flickr-style loader.

Expected input:
  /home/datasets/coco/train2017/*.jpg
  /home/datasets/coco/val2017/*.jpg
  /home/datasets/coco/annotations/captions_train2017.json
  /home/datasets/coco/annotations/captions_val2017.json

Outputs:
  data/coco2017/Images/*.jpg       symlinks to train2017/val2017 images
  data/coco2017/captions.txt       CSV with columns image,caption
  data/coco2017/train2017          symlink to the source folder
  data/coco2017/val2017            symlink to the source folder
  data/coco2017/annotations        symlink to the source folder
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_caption_rows(annotation_path: Path) -> list[dict[str, str]]:
    with annotation_path.open(encoding="utf-8") as f:
        data = json.load(f)

    image_names = {image["id"]: image["file_name"] for image in data["images"]}
    rows = []
    for ann in data["annotations"]:
        file_name = image_names[ann["image_id"]]
        rows.append({"image": file_name, "caption": ann["caption"].strip()})
    return rows


def replace_symlink(target: Path, source: Path) -> None:
    if target.is_symlink():
        if target.resolve() == source.resolve():
            return
        target.unlink()
    elif target.exists():
        return
    target.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def symlink_images(source_dir: Path, images_dir: Path) -> int:
    count = 0
    for image_path in source_dir.glob("*.jpg"):
        target = images_dir / image_path.name
        if target.exists() or target.is_symlink():
            continue
        target.symlink_to(image_path.resolve())
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default="/home/datasets/coco")
    parser.add_argument("--root", "--out-root", default="data/coco2017")
    args = parser.parse_args()

    source_root = Path(args.source_root)
    root = Path(args.root)
    train_ann = source_root / "annotations" / "captions_train2017.json"
    val_ann = source_root / "annotations" / "captions_val2017.json"
    train_images = source_root / "train2017"
    val_images = source_root / "val2017"

    required = [train_ann, val_ann, train_images, val_images]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Missing COCO files/folders:\n" + "\n".join(missing))

    root.mkdir(parents=True, exist_ok=True)
    replace_symlink(root / "train2017", train_images)
    replace_symlink(root / "val2017", val_images)
    replace_symlink(root / "annotations", source_root / "annotations")

    images_dir = root / "Images"
    images_dir.mkdir(parents=True, exist_ok=True)
    linked = symlink_images(train_images, images_dir) + symlink_images(val_images, images_dir)

    rows = load_caption_rows(train_ann) + load_caption_rows(val_ann)
    captions_path = root / "captions.txt"
    with captions_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "caption"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[coco] wrote {captions_path} with {len(rows)} captions")
    print(f"[coco] Images folder: {images_dir} ({linked} new symlinks)")


if __name__ == "__main__":
    main()
