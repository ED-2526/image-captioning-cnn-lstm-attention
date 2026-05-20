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


def train_one_epoch(encoder, decoder, loader, criterion, optimizer, device, p_teacher=1.0):
    """Entrena el model durant una epoch i retorna la loss mitjana.

    p_teacher=1.0 → teacher forcing pur (sempre rep la paraula real)
    p_teacher=0.5 → 50% real, 50% predicció del model (scheduled sampling)
    """
    encoder.train()
    decoder.train()
    total_loss = 0

    for images, captions, lengths in loader:
        images = images.to(device)
        captions = captions.to(device)
        B, T = captions.shape

        features = encoder(images)                      # [B, embed_size]
        states = decoder._init_hidden(features)

        # Generem pas a pas: a cada timestep decidim si usar la paraula real o la predicció
        inp = captions[:, 0]                            # primer input: <start>
        outputs = []
        for t in range(1, T):
            emb = decoder.embed(inp).unsqueeze(1)              # [B, 1, embed]
            out, states = decoder.lstm(emb, states)            # [B, 1, hidden]
            logits = decoder.linear(out.squeeze(1))            # [B, vocab]
            outputs.append(logits)

            # Scheduled sampling: moneda per cada exemple del batch
            use_teacher = torch.rand(B, device=device) < p_teacher
            inp = torch.where(use_teacher, captions[:, t], logits.argmax(dim=1))

        outputs = torch.stack(outputs, dim=1)           # [B, T-1, vocab_size]
        targets = captions[:, 1:]                       # [B, T-1]
        loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))

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
    parser.add_argument("--ss-start", type=float, default=1.0)  # p_teacher inicial
    parser.add_argument("--ss-end",   type=float, default=1.0)  # p_teacher final (1.0 = CE pur)
    parser.add_argument("--topk", type=int, default=0)          # 0 = CrossEntropy normal; >0 = TopK Semantic Loss
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
    train_loader, _, val_ids = get_loaders(
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
    if args.topk > 0 and args.glove_path:
        # TopK Semantic Loss
        # Pas 1: carreguem els vectors GloVe de cada paraula del vocabulari
        glove = load_glove(args.glove_path, vocab, glove_dim=300)  # [V, 300]
        glove = glove / (glove.norm(dim=1, keepdim=True) + 1e-8)   # normalitzem

        # Pas 2: calculem la similitud entre totes les parelles de paraules
        sim = glove @ glove.T   # [V, V]  sim[i,j] = similitud cosinus entre paraula i i j
        sim = sim.to(device)

        def criterion(outputs, targets):
            # outputs: [N, V] logits del model
            # targets: [N]   paraules correctes

            # Pas 3: creem la distribució suau
            # 0.8 a la paraula correcta + 0.2 repartit als k veïns més similars
            soft = torch.zeros_like(outputs)
            soft.scatter_(1, targets.unsqueeze(1), 0.8)             # pes a la correcta
            neighbors = sim[targets].topk(args.topk, dim=1).indices # k veïns de cada paraula
            soft.scatter_(1, neighbors, 0.2 / args.topk)            # pes als veïns

            # Pas 4: cross-entropy amb la distribució suau
            return -(soft * outputs.log_softmax(dim=1)).sum(dim=1).mean()
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=0)
    # Només entrenem el decoder i la capa lineal de l'encoder (ResNet congelat)
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    # 5. Training loop
    best_bleu4 = 0.0
    for epoch in range(1, args.epochs + 1):
        # Scheduled sampling: p_teacher baixa linealment cada epoch
        p_teacher = args.ss_start - (args.ss_start - args.ss_end) * (epoch - 1) / max(args.epochs - 1, 1)
        train_loss = train_one_epoch(encoder, decoder, train_loader, criterion, optimizer, device, p_teacher)

        # Validació: el model genera les captions sol (sense veure les reals)
        bleu1, bleu4, meteor = evaluate_bleu(
            encoder, decoder, vocab, args.captions_csv, val_ids, args.images_dir, device
        )
        print(f"epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  bleu1={bleu1:.3f}  bleu4={bleu4:.3f}  meteor={meteor:.3f}")

        # Guardem el millor model per BLEU-4
        if bleu4 > best_bleu4:
            best_bleu4 = bleu4
            torch.save({
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "args": vars(args),
            }, "checkpoints/best_model.pt")
            print(f"  → nou millor model guardat (bleu4={bleu4:.4f})")


if __name__ == "__main__":
    main()
