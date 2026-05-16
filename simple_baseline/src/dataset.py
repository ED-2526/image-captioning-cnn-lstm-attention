"""Dataset de Flickr8k per image captioning."""

import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from vocabulary import Vocabulary


def get_transform(train=True):
    """Transformacions estàndard per a ResNet.

    Durant training: random crop i flip horitzontal (augmentació).
    Durant validació: només el necessari per ResNet (resize + normalize).
    """
    if train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


class Flickr8kDataset(Dataset):
    """Cada mostra és un parell (imatge, caption).

    Si una imatge té 5 captions, apareix 5 vegades al dataset.
    """

    def __init__(self, images_dir, captions_csv, vocab, image_ids=None, train=True):
        self.images_dir = images_dir
        self.vocab = vocab
        self.transform = get_transform(train)

        # Llegim el CSV i filtrem per image_ids si cal
        df = pd.read_csv(captions_csv)
        if image_ids is not None:
            df = df[df["image"].isin(image_ids)].reset_index(drop=True)
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["image"]
        caption = str(row["caption"])

        # Obrim la imatge i apliquem transformacions → tensor [3, 224, 224]
        image = Image.open(os.path.join(self.images_dir, img_name)).convert("RGB")
        image = self.transform(image)

        # Convertim la caption a tensor d'enters: 'a dog' → [1, 4, 27, 2]
        ids = self.vocab.encode(caption)
        caption_tensor = torch.tensor(ids, dtype=torch.long)

        return image, caption_tensor


def collate_fn(batch):
    """Ajuntem mostres de diferent longitud en un batch.

    Com les captions tenen longituds diferents, fem padding amb zeros (<pad>)
    fins que totes tinguin la mateixa longitud (la de la més llarga del batch).
    """
    images, captions = zip(*batch)

    # Stack de les imatges → [B, 3, 224, 224]
    images = torch.stack(images)

    # Padding de les captions → [B, T_max]
    max_len = max(len(c) for c in captions)
    padded = torch.zeros(len(captions), max_len, dtype=torch.long)
    for i, cap in enumerate(captions):
        padded[i, :len(cap)] = cap

    # Longituds reals de cada caption (per saber fins on és text real)
    lengths = torch.tensor([len(c) for c in captions])

    return images, padded, lengths


def get_loaders(images_dir, captions_csv, vocab, batch_size=32, val_size=1000):
    """Crea els DataLoaders de train i validació.

    Dividim les imatges (no les captions) en train i val.
    Usem seed fixe per reproduïbilitat.
    """
    import random
    random.seed(42)

    # Agafem els IDs únics de les imatges i els barregem
    all_ids = pd.read_csv(captions_csv)["image"].unique().tolist()
    random.shuffle(all_ids)

    val_ids = set(all_ids[:val_size])
    train_ids = set(all_ids[val_size:])

    train_ds = Flickr8kDataset(images_dir, captions_csv, vocab, train_ids, train=True)
    val_ds = Flickr8kDataset(images_dir, captions_csv, vocab, val_ids, train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=2)

    return train_loader, val_loader, list(val_ids)
