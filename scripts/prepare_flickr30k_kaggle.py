#!/usr/bin/env python3
"""Prepare the Kaggle Flickr30k dump for the local Flickr-style loader.

Expected input after unzipping awsaf49/flickr30k-dataset:
  data/flickr30k/**/results.csv
  data/flickr30k/**/<image files>.jpg

Outputs:
  data/flickr30k_hf/Images       symlink to the image folder when possible
  data/flickr30k_hf/captions.txt CSV with columns image,caption
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def count_jpgs(path: Path) -> int:
    return sum(1 for _ in path.glob("*.jpg"))


def find_images_dir(root: Path) -> Path:
    candidates = [
        root / "flickr30k_images",
        root / "flickr30k_images" / "flickr30k_images",
        root / "Images",
    ]
    candidates.extend(path for path in root.rglob("*") if path.is_dir())

    best_dir: Path | None = None
    best_count = 0
    for path in candidates:
        if not path.exists() or not path.is_dir():
            continue
        jpg_count = count_jpgs(path)
        if jpg_count > best_count:
            best_dir = path
            best_count = jpg_count

    if best_dir is None or best_count == 0:
        raise SystemExit(f"Could not find a Flickr30k image directory under {root}")
    return best_dir


def find_captions_file(root: Path) -> Path:
    preferred = [
        root / "results.csv",
        root / "captions.txt",
        root / "flickr30k_images" / "results.csv",
        root / "flickr30k_images" / "captions.txt",
    ]
    candidates = [path for path in preferred if path.exists()]
    candidates.extend(root.rglob("results.csv"))
    candidates.extend(root.rglob("captions.txt"))

    for path in candidates:
        with path.open(encoding="utf-8", errors="replace") as f:
            header = f.readline().strip()
        normalized = header.replace(" ", "")
        if "image_name" in normalized and "comment" in normalized:
            return path
        if "image" in normalized and "caption" in normalized:
            return path

    raise SystemExit(f"Could not find results.csv/captions.txt under {root}")


def read_caption_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", errors="replace", newline="") as f:
        header = f.readline()
        delimiter = "|" if "|" in header else ","
        f.seek(0)
        reader = csv.DictReader(f, delimiter=delimiter, skipinitialspace=True)
        if reader.fieldnames is None:
            raise SystemExit(f"No CSV header found in {path}")

        field_by_name = {field.strip(): field for field in reader.fieldnames}
        image_key = field_by_name.get("image") or field_by_name.get("image_name")
        caption_key = field_by_name.get("caption") or field_by_name.get("comment")
        if image_key is None or caption_key is None:
            raise SystemExit(f"Unsupported captions columns in {path}: {reader.fieldnames}")

        rows = []
        for row in reader:
            image = (row.get(image_key) or "").strip()
            caption = (row.get(caption_key) or "").strip()
            if image and caption:
                rows.append({"image": image, "caption": caption})
    return rows


def link_or_populate_images(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.is_symlink():
        if target.resolve() == source.resolve():
            return f"Images symlink already points to {source}"
        raise SystemExit(f"{target} is a symlink to {target.resolve()}, not {source}")

    if target.exists():
        if not target.is_dir():
            raise SystemExit(f"{target} exists and is not a directory")
        created = 0
        for image_path in source.glob("*.jpg"):
            link = target / image_path.name
            if not link.exists() and not link.is_symlink():
                link.symlink_to(image_path.resolve())
                created += 1
        return f"Images directory exists; added {created} image symlinks"

    target.symlink_to(source.resolve(), target_is_directory=True)
    return f"Created Images symlink -> {source}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/flickr30k")
    parser.add_argument("--out", default="data/flickr30k_hf")
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    if not root.exists():
        raise SystemExit(f"Missing Flickr30k root: {root}")

    images_dir = find_images_dir(root)
    captions_file = find_captions_file(root)
    rows = read_caption_rows(captions_file)

    out.mkdir(parents=True, exist_ok=True)
    captions_out = out / "captions.txt"
    with captions_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "caption"])
        writer.writeheader()
        writer.writerows(rows)

    link_message = link_or_populate_images(images_dir, out / "Images")

    available_images = {path.name for path in images_dir.glob("*.jpg")}
    caption_images = {row["image"] for row in rows}
    missing = sorted(caption_images - available_images)
    if missing:
        preview = ", ".join(missing[:5])
        raise SystemExit(f"{len(missing)} caption images are missing from {images_dir}: {preview}")

    print(f"[flickr30k] source images: {images_dir} ({len(available_images)} jpgs)")
    print(f"[flickr30k] source captions: {captions_file}")
    print(f"[flickr30k] wrote {captions_out} with {len(rows)} captions")
    print(f"[flickr30k] {link_message}")


if __name__ == "__main__":
    main()
