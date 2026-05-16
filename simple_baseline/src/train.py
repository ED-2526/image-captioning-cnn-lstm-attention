"""Training script per image captioning (CNN + LSTM).

Ús:
    python train.py --images-dir /path/images --captions-csv /path/captions.txt
"""

import argparse
import pickle
from pathlib import Path

import nltk
import pandas as pd
import torch
import torch.nn as nn
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

from vocabulary import build_vocab, tokenize, load_glove
from dataset import get_loaders
from model import EncoderCNN, DecoderLSTM


def train_one_epoch(encoder, decoder, loader, criterion, optimizer, device):
    """Entrena el model durant una epoch i retorna la loss mitjana."""
    encoder.train()
    decoder.train()
    total_loss = 0

    for images, captions, lengths in loader:
        images = images.to(device)
        captions = captions.to(device)

        # --- Forward pass ---
        # 1. Encoder: imatge → vector de característiques
        features = encoder(images)                # [B, embed_size]

        # 2. Decoder: features + captions[:-1] → prediccions
        # Li donem tots els tokens menys l'últim com a input
        # i esperem tots els tokens menys el primer com a target
        inputs = captions[:, :-1]                 # [B, T-1] - input: <start>...paraules
        targets = captions[:, 1:]                 # [B, T-1] - target: paraules...<end>

        outputs = decoder(features, inputs)       # [B, T-1, vocab_size]

        # 3. Loss: comparem prediccions amb targets
        # Aplanem per CrossEntropy: [B*T-1, vocab_size] vs [B*T-1]
        B, T, V = outputs.shape
        loss = criterion(outputs.reshape(B * T, V), targets.reshape(B * T))

        # --- Backward pass ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate_bleu(encoder, decoder, vocab, captions_csv, val_ids, images_dir, device):
    """Calcula BLEU-1, BLEU-4 i METEOR sobre el val set.

    Per a cada imatge del val set:
    1. Generem una caption amb greedy decoding
    2. Comparem amb les 5 captions reals de referència
    """
    encoder.eval()
    decoder.eval()

    df = pd.read_csv(captions_csv)
    smooth = SmoothingFunction().method1

    references_all = []  # llista de llistes de captions reals (tokenitzades)
    hypotheses_all = []  # llista de captions generades (tokenitzades)

    from dataset import get_transform
    from PIL import Image

    transform = get_transform(train=False)

    for img_name in val_ids:
        # Captions reals d'aquesta imatge (normalment 5)
        refs = df[df["image"] == img_name]["caption"].tolist()
        refs_tok = [tokenize(r) for r in refs]

        # Generem caption amb el model
        img = Image.open(Path(images_dir) / img_name).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        features = encoder(x)
        ids = decoder.sample(features)[0].cpu().tolist()
        hyp = tokenize(vocab.decode(ids))

        if len(hyp) == 0:
            continue

        references_all.append(refs_tok)
        hypotheses_all.append(hyp)

    # BLEU-1 i BLEU-4
    bleu1 = corpus_bleu(references_all, hypotheses_all,
                        weights=(1, 0, 0, 0), smoothing_function=smooth)
    bleu4 = corpus_bleu(references_all, hypotheses_all,
                        weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

    # METEOR (mitjana sobre totes les imatges)
    meteor = sum(
        meteor_score(refs, hyp)
        for refs, hyp in zip(references_all, hypotheses_all)
    ) / len(hypotheses_all)

    return bleu1, bleu4, meteor


@torch.no_grad()
def evaluate(encoder, decoder, loader, criterion, device):
    """Calcula la loss de validació (sense actualitzar pesos)."""
    encoder.eval()
    decoder.eval()
    total_loss = 0

    for images, captions, lengths in loader:
        images = images.to(device)
        captions = captions.to(device)

        features = encoder(images)
        inputs = captions[:, :-1]
        targets = captions[:, 1:]
        outputs = decoder(features, inputs)
        #batch size, long_max, vocab size
        B, T, V = outputs.shape
        loss = criterion(outputs.reshape(B * T, V), targets.reshape(B * T))
        total_loss += loss.item()

    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True, help="Directori amb les imatges")
    parser.add_argument("--captions-csv", required=True, help="CSV amb columnes image,caption")
    parser.add_argument("--encoder-size", type=int, default=256)
    parser.add_argument("--word-embed-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--glove-path", default=None, help="Ruta al fitxer GloVe (.txt). Opcional.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--vocab-threshold", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # 1. Construïm el vocabulari
    print("[vocab] construint...")
    vocab = build_vocab(args.captions_csv, threshold=args.vocab_threshold)
    print(f"[vocab] mida = {len(vocab)}")

    # Guardem el vocabulari per poder-lo usar a inferència
    Path("checkpoints").mkdir(exist_ok=True)
    with open("checkpoints/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    # 2. Creem els DataLoaders
    train_loader, val_loader, val_ids = get_loaders(
        args.images_dir, args.captions_csv, vocab, args.batch_size
    )
    print(f"[data] train={len(train_loader)} batches  val={len(val_loader)} batches")

    # 3. Creem el model
    # Si s'especifica GloVe, els word embeddings tindran la seva dimensió (300)
    glove_weights = None
    if args.glove_path:
        glove_weights = load_glove(args.glove_path, vocab, glove_dim=args.word_embed_size)

    encoder = EncoderCNN(encoder_size=args.encoder_size).to(device)
    decoder = DecoderLSTM(
        encoder_size=args.encoder_size,
        word_embed_size=args.word_embed_size,
        hidden_size=args.hidden_size,
        vocab_size=len(vocab),
        pretrained_embeddings=glove_weights,
    ).to(device)

    # 4. Loss i optimizer
    # ignore_index=0 → no calculem loss sobre els tokens <pad>
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    # Només entrenem el decoder i la capa lineal de l'encoder (ResNet congelat)
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    # 5. Training loop
    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(encoder, decoder, train_loader, criterion, optimizer, device)
        val_loss = evaluate(encoder, decoder, val_loader, criterion, device)

        bleu1, bleu4, meteor = evaluate_bleu(
            encoder, decoder, vocab, args.captions_csv, val_ids, args.images_dir, device
        )
        print(f"epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  bleu1={bleu1:.3f}  bleu4={bleu4:.3f}  meteor={meteor:.3f}")

        # Guardem el millor model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "args": vars(args),
            }, "checkpoints/best_model.pt")
            print(f"  → nou millor model guardat (val_loss={val_loss:.4f})")


if __name__ == "__main__":
    main()
