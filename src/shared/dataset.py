"""Dataset and DataLoader utilities for Flickr-style image captioning."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.shared.vocabulary import Vocabulary


def load_captions_df(captions_csv: str | Path) -> pd.DataFrame:
    """Load Flickr8k or Flickr30k captions into `image` and `caption` columns."""
    captions_csv = Path(captions_csv)
    with open(captions_csv, encoding="utf-8") as f:
        header = f.readline()

    if "|" in header:
        df = pd.read_csv(captions_csv, sep="|", skipinitialspace=True)
        df.columns = [column.strip() for column in df.columns]
        df = df.rename(columns={"image_name": "image", "comment": "caption"})
        df = df[["image", "caption"]].copy()
        df["image"] = df["image"].str.strip()
        df["caption"] = df["caption"].astype(str).str.strip()
    else:
        df = pd.read_csv(captions_csv)
        df = df[["image", "caption"]].copy()

    return df.reset_index(drop=True)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_transform(image_size: int = 224, train: bool = True):
    if train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class Flickr8kDataset(Dataset):
    """Return `(image_tensor, caption_ids_tensor)` samples from a captions CSV."""

    def __init__(
        self,
        images_dir: str | Path,
        captions_csv: str | Path,
        vocab: Vocabulary,
        transform=None,
        image_ids: list[str] | None = None,
        return_image_id: bool = False,
    ):
        self.images_dir = Path(images_dir)
        self.vocab = vocab
        self.transform = transform if transform is not None else get_transform(train=False)
        self.return_image_id = return_image_id

        df = load_captions_df(captions_csv)
        if image_ids is not None:
            df = df[df["image"].isin(set(image_ids))].reset_index(drop=True)
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_name = row["image"]
        caption = str(row["caption"])

        image = Image.open(self.images_dir / img_name).convert("RGB")
        image = self.transform(image)
        ids = self.vocab.encode(caption, add_special=True)

        if self.return_image_id:
            return image, torch.tensor(ids, dtype=torch.long), img_name
        return image, torch.tensor(ids, dtype=torch.long)


def collate_fn(batch):
    """Sort captions by length and pad them to the longest sequence in the batch."""
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*batch)
    images = torch.stack(images, dim=0)

    lengths = [len(caption) for caption in captions]
    targets = torch.zeros(len(captions), max(lengths), dtype=torch.long)
    for i, caption in enumerate(captions):
        targets[i, : lengths[i]] = caption
    return images, targets, lengths


def split_image_ids(
    captions_csv: str | Path,
    val_size: int = 1000,
    test_size: int = 1000,
    seed: int = 42,
):
    """Split unique image filenames into train/validation/test groups."""
    import numpy as np

    df = load_captions_df(captions_csv)
    unique = sorted(df["image"].unique().tolist())
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)

    test = unique[:test_size]
    val = unique[test_size : test_size + val_size]
    train = unique[test_size + val_size :]
    return train, val, test


def get_loaders(
    images_dir: str | Path,
    captions_csv: str | Path,
    vocab: Vocabulary,
    batch_size: int = 32,
    num_workers: int = 2,
    image_size: int = 224,
):
    train_ids, val_ids, test_ids = split_image_ids(captions_csv)

    train_ds = Flickr8kDataset(
        images_dir,
        captions_csv,
        vocab,
        transform=get_transform(image_size, train=True),
        image_ids=train_ids,
    )
    val_ds = Flickr8kDataset(
        images_dir,
        captions_csv,
        vocab,
        transform=get_transform(image_size, train=False),
        image_ids=val_ids,
    )
    test_ds = Flickr8kDataset(
        images_dir,
        captions_csv,
        vocab,
        transform=get_transform(image_size, train=False),
        image_ids=test_ids,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, (train_ids, val_ids, test_ids)


class Flickr30kHFDataset(Dataset):
    """Expand HuggingFace Flickr30k rows into one sample per caption."""

    def __init__(
        self,
        hf_split,
        vocab: Vocabulary,
        transform=None,
        return_image_id: bool = False,
    ):
        self.vocab = vocab
        self.transform = transform if transform is not None else get_transform(train=False)
        self.return_image_id = return_image_id
        self.samples: list[tuple[int, int]] = []
        self.hf_data = hf_split

        for img_idx in range(len(hf_split)):
            for cap_idx in range(len(hf_split[img_idx]["caption"])):
                self.samples.append((img_idx, cap_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_idx, cap_idx = self.samples[idx]
        row = self.hf_data[img_idx]

        image = row["image"].convert("RGB")
        image = self.transform(image)
        ids = self.vocab.encode(row["caption"][cap_idx], add_special=True)

        if self.return_image_id:
            return image, torch.tensor(ids, dtype=torch.long), row["filename"]
        return image, torch.tensor(ids, dtype=torch.long)


def get_loaders_hf(
    hf_dataset,
    vocab: Vocabulary,
    batch_size: int = 32,
    num_workers: int = 2,
    image_size: int = 224,
):
    """Create train/validation/test loaders from `nlphuji/flickr30k`."""
    full = hf_dataset["test"]

    train_hf = full.filter(lambda x: x["split"] == "train")
    val_hf = full.filter(lambda x: x["split"] == "val")
    test_hf = full.filter(lambda x: x["split"] == "test")

    train_ds = Flickr30kHFDataset(train_hf, vocab, get_transform(image_size, train=True))
    val_ds = Flickr30kHFDataset(val_hf, vocab, get_transform(image_size, train=False))
    test_ds = Flickr30kHFDataset(test_hf, vocab, get_transform(image_size, train=False))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


class COCODataset(Dataset):
    """COCO captions dataset returning one image/caption pair per annotation."""

    def __init__(
        self,
        images_dir: str | Path,
        annotations_json: str | Path,
        vocab: Vocabulary,
        transform=None,
        max_images: int | None = None,
    ):
        import json

        self.images_dir = Path(images_dir)
        self.vocab = vocab
        self.transform = transform if transform is not None else get_transform(train=False)

        with open(annotations_json, encoding="utf-8") as f:
            data = json.load(f)

        id_to_file = {image["id"]: image["file_name"] for image in data["images"]}
        allowed_ids = set(id_to_file)
        if max_images is not None:
            allowed_ids = set(list(id_to_file)[:max_images])

        self.samples = [
            (id_to_file[ann["image_id"]], ann["caption"])
            for ann in data["annotations"]
            if ann["image_id"] in allowed_ids
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_name, caption = self.samples[idx]
        image = Image.open(self.images_dir / img_name).convert("RGB")
        image = self.transform(image)
        ids = self.vocab.encode(caption, add_special=True)
        return image, torch.tensor(ids, dtype=torch.long)


def get_loaders_coco(
    coco_dir: str | Path,
    vocab: Vocabulary,
    batch_size: int = 32,
    num_workers: int = 2,
    image_size: int = 224,
    max_train_images: int | None = None,
):
    """Create train/validation loaders for COCO 2017 captions."""
    coco_dir = Path(coco_dir)
    train_ds = COCODataset(
        coco_dir / "train2017",
        coco_dir / "annotations/captions_train2017.json",
        vocab,
        transform=get_transform(image_size, train=True),
        max_images=max_train_images,
    )
    val_ds = COCODataset(
        coco_dir / "val2017",
        coco_dir / "annotations/captions_val2017.json",
        vocab,
        transform=get_transform(image_size, train=False),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return train_loader, val_loader
