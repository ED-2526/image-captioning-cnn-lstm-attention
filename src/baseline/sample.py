"""Generate a caption for a single image using a trained checkpoint."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch
from PIL import Image

from src.baseline.model import DecoderRNN, EncoderCNN
from src.shared.dataset import get_transform
from src.shared.vocabulary import Vocabulary  # noqa: F401


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(ckpt_path: str, vocab_path: str, device: torch.device):
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt["args"]
    pretrained = ckpt.get("checkpoint_mode") == "light" and not args.get("no_pretrained_backbone", False)
    encoder = EncoderCNN(args["embed_size"], backbone=args["backbone"], pretrained=pretrained).to(device).eval()
    decoder = DecoderRNN(
        args["embed_size"],
        args["hidden_size"],
        len(vocab),
        args["num_layers"],
        dropout=args.get("dropout", 0.5),
        decoder_direction=args.get("decoder_direction", "uni"),
    ).to(device).eval()
    encoder.load_state_dict(ckpt["encoder"], strict=ckpt.get("checkpoint_mode", "full") == "full")
    decoder.load_state_dict(ckpt["decoder"])
    return encoder, decoder, vocab


@torch.no_grad()
def caption_image(image_path: str, encoder, decoder, vocab, device) -> str:
    tfm = get_transform(train=False)
    img = Image.open(image_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    feat = encoder(x)
    ids = decoder.sample(feat).cpu().numpy()[0].tolist()
    return vocab.decode(ids)


@torch.no_grad()
def caption_pil_image(pil_img, encoder, decoder, vocab, device) -> str:
    tfm = get_transform(train=False)
    x = tfm(pil_img.convert("RGB")).unsqueeze(0).to(device)
    feat = encoder(x)
    ids = decoder.sample(feat).cpu().numpy()[0].tolist()
    return vocab.decode(ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab", default="data/flickr8k/vocab.pkl")
    args = parser.parse_args()

    device = detect_device()
    encoder, decoder, vocab = load_checkpoint(args.checkpoint, args.vocab, device)
    cap = caption_image(args.image, encoder, decoder, vocab, device)
    print(f"{Path(args.image).name}: {cap}")


if __name__ == "__main__":
    main()
