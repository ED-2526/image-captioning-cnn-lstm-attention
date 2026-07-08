"""Training script for the Attention-based Image Captioning model.

Usage:
    python -m src.attention.train --epochs 10 --batch-size 32
"""
from __future__ import annotations

import argparse
import contextlib
import pickle
import time
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

from src.shared.dataset import get_loaders, split_image_ids, load_captions_df, get_loaders_hf
from src.attention.model import AttentionDecoder, EncoderCNNAttention
from src.attention.sample import caption_image, caption_pil_image
from src.shared.vocabulary import Vocabulary, build_vocab, build_vocab_hf, simple_tokenize, load_glove_weights, load_word2vec_weights
from src.shared.losses import SemanticCrossEntropyLoss, build_soft_labels


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", default="dataset/Images")
    p.add_argument("--captions-csv", default="dataset/captions.txt")
    p.add_argument("--vocab-path", default="dataset/vocab.pkl")
    p.add_argument("--checkpoints-dir", default="checkpoints_attention")
    p.add_argument("--vocab-threshold", type=int, default=5)

    p.add_argument("--embed-size", type=int, default=256)
    p.add_argument("--hidden-size", type=int, default=512)
    p.add_argument("--attention-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--backbone", default="resnet152")

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=999)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--log-step", type=int, default=20)

    p.add_argument("--glove-path", default=None,
                   help="Path to a GloVe text file used to initialize embeddings.")
    p.add_argument("--word2vec-path", default=None,
                   help="Path to a Word2Vec .bin or .txt file. Ignored when --glove-path is set.")
    p.add_argument("--word2vec-binary", action="store_true",
                   help="Treat the Word2Vec file as binary.")
    p.add_argument("--no-semantic-loss", action="store_true",
                   help="Use standard CrossEntropyLoss even with pretrained embeddings.")
    p.add_argument("--freeze-embeddings", action="store_true",
                   help="Freeze pretrained word embeddings during training.")
    p.add_argument("--semantic-temp", type=float, default=10.0,
                   help="Temperature for semantic soft labels.")
    p.add_argument("--finetune-cnn-epoch", type=int, default=None,
                   help="Epoch after which ResNet layer4 is unfrozen.")
    p.add_argument("--ds-lambda", type=float, default=1.0,
                   help="Doubly stochastic attention regularization weight.")
    p.add_argument("--label-smoothing", type=float, default=0.0,
                   help="Label smoothing for CrossEntropyLoss. Use 0.0 to disable.")
    p.add_argument("--resume-from", default=None,
                   help="Checkpoint used to resume training.")

    # ── Flickr30k HuggingFace ──────────────────────────────────────────────
    p.add_argument("--flickr30k-hf", action="store_true",
                   help="Use the HuggingFace Flickr30k dataset instead of local CSV files.")
    p.add_argument("--flickr30k-hf-cache", default="dataset/flickr30k_hf",
                   help="HuggingFace dataset cache directory.")

    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="image-captioning")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--run-name", default=None)

    return p.parse_args()


def safe_save(obj, path, retries: int = 5):
    """torch.save amb reintents per errors NFS transitòris."""
    import shutil, time
    path = Path(path)
    for attempt in range(retries):
        try:
            tmp = path.parent / f".tmp_{path.name}"
            torch.save(obj, tmp)
            shutil.move(str(tmp), str(path))
            return
        except RuntimeError:
            if attempt < retries - 1:
                print(f"[ckpt] error NFS (intent {attempt+1}/{retries}), reintentant...")
                time.sleep(3)
            else:
                raise


def get_or_build_vocab(args) -> Vocabulary:
    vp = Path(args.vocab_path)
    if vp.exists():
        with open(vp, "rb") as f:
            return pickle.load(f)
    vocab = build_vocab(args.captions_csv, threshold=args.vocab_threshold)
    vp.parent.mkdir(parents=True, exist_ok=True)
    with open(vp, "wb") as f:
        pickle.dump(vocab, f)
    print(f"[vocab] built and saved to {vp} (size={len(vocab)})")
    return vocab


@torch.no_grad()
def evaluate(encoder, decoder, loader, criterion, device) -> float:
    encoder.eval()
    decoder.eval()
    losses = []
    for images, captions, lengths in loader:
        images = images.to(device)
        captions = captions.to(device)
        targets = pack_padded_sequence(
            captions[:, 1:], [l - 1 for l in lengths], batch_first=True
        ).data
        features = encoder(images)
        outputs, _ = decoder(features, captions, lengths)
        loss = criterion(outputs, targets)
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def evaluate_bleu(encoder, decoder, vocab, val_ids, df_caps, test_pil, args, device) -> dict:
    """Calcula BLEU-1, BLEU-4 i METEOR sobre el val set. Retorna dict amb les mètriques."""
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    encoder.eval()
    decoder.eval()
    smooth = SmoothingFunction().method1
    all_refs, all_hyps, all_meteors = [], [], []
    for img in val_ids:
        refs = [simple_tokenize(c) for c in df_caps[df_caps["image"] == img]["caption"].tolist()]
        if not refs:
            continue
        if args.flickr30k_hf:
            hyp = simple_tokenize(caption_pil_image(test_pil[img], encoder, decoder, vocab, device))
        else:
            hyp = simple_tokenize(caption_image(f"{args.images_dir}/{img}", encoder, decoder, vocab, device))
        all_refs.append(refs)
        all_hyps.append(hyp)
        all_meteors.append(meteor_score(refs, hyp))
    b1 = corpus_bleu(all_refs, all_hyps, weights=(1, 0, 0, 0))
    b4 = corpus_bleu(all_refs, all_hyps, weights=(.25, .25, .25, .25))
    m  = float(np.mean(all_meteors))
    return {"val/bleu1": b1, "val/bleu4": b4, "val/meteor": m}


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    Path(args.checkpoints_dir).mkdir(parents=True, exist_ok=True)

    # ── Carrega dataset i vocab (CSV o HuggingFace) ────────────────────────
    if args.flickr30k_hf:
        from datasets import load_dataset
        print("[data] carregant Flickr30k HuggingFace...")
        hf_ds = load_dataset("nlphuji/flickr30k", trust_remote_code=True,
                             cache_dir=args.flickr30k_hf_cache)
        vp = Path(args.vocab_path)
        if vp.exists():
            with open(vp, "rb") as f:
                import pickle; vocab = pickle.load(f)
            print(f"[vocab] carregat de {vp} (size={len(vocab)})")
        else:
            print("[vocab] construint des de HF dataset...")
            vocab = build_vocab_hf(hf_ds, threshold=args.vocab_threshold)
            vp.parent.mkdir(parents=True, exist_ok=True)
            with open(vp, "wb") as f:
                import pickle; pickle.dump(vocab, f)
            print(f"[vocab] built and saved to {vp} (size={len(vocab)})")

        train_loader, val_loader, _ = get_loaders_hf(
            hf_ds, vocab, batch_size=args.batch_size, num_workers=args.num_workers)

        # Construeix train_ids / test_ids / df_caps per a BLEU
        full = hf_ds["test"]
        train_rows = full.filter(lambda x: x["split"] == "train")
        val_rows   = full.filter(lambda x: x["split"] == "val")
        test_rows  = full.filter(lambda x: x["split"] == "test")
        train_ids  = [r["filename"] for r in train_rows]
        val_ids    = [r["filename"] for r in val_rows]
        test_ids   = [r["filename"] for r in test_rows]
        # df_caps: una fila per (imatge, caption) — igual que el CSV
        records = []
        for r in full:
            for cap in r["caption"]:
                records.append({"image": r["filename"], "caption": cap})
        import pandas as _pd
        df_caps_hf = _pd.DataFrame(records)
        # dict filename → PIL image per al test i val (per BLEU sense path a disc)
        test_pil = {r["filename"]: r["image"] for r in test_rows}
        val_pil  = {r["filename"]: r["image"] for r in val_rows}
    else:
        vocab = get_or_build_vocab(args)
        train_loader, val_loader, _, (train_ids, val_ids, test_ids) = get_loaders(
            images_dir=args.images_dir,
            captions_csv=args.captions_csv,
            vocab=vocab,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    print(f"[vocab] size = {len(vocab)}")
    print(f"[data] train batches={len(train_loader)}  val batches={len(val_loader)}")

    encoder = EncoderCNNAttention(backbone=args.backbone).to(device)

    pretrained_weights = None
    if args.glove_path:
        pretrained_weights, glove_dim = load_glove_weights(args.glove_path, vocab)
        pretrained_weights = pretrained_weights.to(device)
        args.embed_size = glove_dim
        emb_type = "glove-frozen" if args.freeze_embeddings else "glove-finetune"
    elif args.word2vec_path:
        binary = args.word2vec_binary if args.word2vec_binary else None
        pretrained_weights, w2v_dim = load_word2vec_weights(args.word2vec_path, vocab, binary=binary)
        pretrained_weights = pretrained_weights.to(device)
        args.embed_size = w2v_dim
        emb_type = "word2vec-frozen" if args.freeze_embeddings else "word2vec-finetune"
    else:
        emb_type = "scratch"
    print(f"[embeddings] tipus={emb_type}  embed_size={args.embed_size}")

    decoder = AttentionDecoder(
        encoder_dim=encoder.encoder_dim,
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        vocab_size=len(vocab),
        attention_dim=args.attention_dim,
        dropout=args.dropout,
        pretrained_weights=pretrained_weights,
        freeze_embeddings=args.freeze_embeddings,
    ).to(device)

    if pretrained_weights is not None and not args.no_semantic_loss:
        soft_lbls = build_soft_labels(decoder.embed.weight.data.cpu(), temperature=args.semantic_temp)
        criterion = SemanticCrossEntropyLoss(soft_lbls).to(device)
        print(f"[loss] SemanticCrossEntropy (temp={args.semantic_temp}) — soft labels des de {emb_type}")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        ls_tag = f" (label_smoothing={args.label_smoothing})" if args.label_smoothing > 0 else ""
        print(f"[loss] CrossEntropyLoss estàndard{ls_tag}")
    optimizer = torch.optim.Adam(
        list(decoder.parameters()), lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5
    )

    use_wandb = args.wandb
    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                   name=args.run_name, config=vars(args))
        wandb.config.update({"vocab_size": len(vocab)})

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    global_step = 0
    start_epoch = 1

    if args.resume_from:
        print(f"[resume] carregant checkpoint: {args.resume_from}")
        res = torch.load(args.resume_from, map_location=device, weights_only=False)
        encoder.load_state_dict(res["encoder"])
        decoder.load_state_dict(res["decoder"])
        start_epoch = res["epoch"] + 1
        print(f"[resume] continuant des de l'epoch {res['epoch']} → inici epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        if args.finetune_cnn_epoch and epoch == args.finetune_cnn_epoch:
            encoder.finetuning = True
            for p in encoder.net.layer4.parameters():
                p.requires_grad = True
            optimizer.add_param_group({"params": list(encoder.net.layer4.parameters()), "lr": args.lr / 10})
            print(f"[finetune] epoch {epoch}: layer4 descongelada (lr={args.lr/10:.2e})")

        encoder.train()
        decoder.train()
        t0 = time.time()
        epoch_losses = []

        for i, (images, captions, lengths) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            captions = captions.to(device, non_blocking=True)
            targets = pack_padded_sequence(
                captions[:, 1:], [l - 1 for l in lengths], batch_first=True
            ).data

            features = encoder(images)
            outputs, alphas_sum = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            if args.ds_lambda > 0:
                loss = loss + args.ds_lambda * ((1 - alphas_sum) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5.0)
            optimizer.step()

            global_step += 1
            epoch_losses.append(loss.item())
            train_losses.append(loss.item())
            if i % args.log_step == 0:
                ppl = float(np.exp(min(loss.item(), 20)))
                print(f"epoch {epoch}/{args.epochs}  step {i}/{len(train_loader)}  "
                      f"loss={loss.item():.4f}  ppl={ppl:.2f}")

        train_loss_epoch = float(np.mean(epoch_losses))
        train_ppl_epoch = float(np.exp(min(train_loss_epoch, 20)))
        val_loss = evaluate(encoder, decoder, val_loader, criterion, device)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        val_ppl = float(np.exp(min(val_loss, 20)))
        elapsed = time.time() - t0
        _df_caps_eval = df_caps_hf if args.flickr30k_hf else load_captions_df(args.captions_csv)
        _val_pil_eval = val_pil if args.flickr30k_hf else None
        bleu_metrics = evaluate_bleu(encoder, decoder, vocab, val_ids, _df_caps_eval, _val_pil_eval, args, device)
        print(f"== epoch {epoch} done  train_loss={train_loss_epoch:.4f}  val_loss={val_loss:.4f}  "
              f"bleu4={bleu_metrics['val/bleu4']:.3f}  meteor={bleu_metrics['val/meteor']:.3f}  ({elapsed:.0f}s)")
        if use_wandb:
            wandb.log({
                "train/loss": train_loss_epoch,
                "train/perplexity": train_ppl_epoch,
                "val/loss": val_loss,
                "val/perplexity": val_ppl,
                **bleu_metrics,
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
            })

        ckpt = {
            "epoch": epoch,
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "vocab_size": len(vocab),
            "args": vars(args),
        }
        out = Path(args.checkpoints_dir) / f"ckpt_epoch{epoch}.pt"
        safe_save(ckpt, out)
        print(f"[ckpt] saved {out}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            safe_save(ckpt, Path(args.checkpoints_dir) / "ckpt_best.pt")
            print(f"[best] new best val_loss={best_val_loss:.4f} -> saved ckpt_best.pt")

    if train_losses:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(range(1, len(train_losses) + 1), train_losses, marker="o", label="train")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("cross-entropy loss")
        axes[0].set_title("Train loss per epoch (attention)")
        axes[0].legend()
        axes[1].plot(range(1, len(val_losses) + 1), val_losses, marker="o", label="val")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("cross-entropy loss")
        axes[1].set_title("Val loss per epoch (attention)")
        axes[1].legend()
        plt.tight_layout()
        plot_path = Path(args.checkpoints_dir) / "loss_curve.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"[plot] saved {plot_path}")

    best_ckpt_path = Path(args.checkpoints_dir) / "ckpt_best.pt"

    # --- BLEU + METEOR evaluation on test set (millor checkpoint) ---
    import nltk
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    print("\n[bleu+meteor] avaluant sobre el conjunt de test...")
    best_ckpt = torch.load(best_ckpt_path, map_location=device)
    encoder.load_state_dict(best_ckpt["encoder"])  # carrega pesos del millor model
    decoder.load_state_dict(best_ckpt["decoder"])
    encoder.eval()
    decoder.eval()

    if args.flickr30k_hf:
        df_caps = df_caps_hf
    else:
        df_caps = load_captions_df(args.captions_csv)  # llegeix totes les captions (Flickr8k o Flickr30k)
    smooth = SmoothingFunction().method1  # suavitzat per evitar BLEU-4 = 0

    all_refs, all_hyps = [], []
    all_meteors = []
    bleu_table = wandb.Table(columns=["image", "generated_caption", "reference_captions", "BLEU-1", "BLEU-4", "METEOR"]) if use_wandb else None
    images_dir_abs = Path(args.images_dir).resolve()
    TABLE_LIMIT = 200

    print(f"Evaluating {len(test_ids)} test images...")
    print(f"{'Image':<35} {'BLEU-1':>7} {'BLEU-4':>7} {'METEOR':>7}  Caption")
    print("-" * 110)
    for img in test_ids:
        refs = [simple_tokenize(c) for c in df_caps[df_caps["image"] == img]["caption"].tolist()]
        if args.flickr30k_hf:
            hyp = simple_tokenize(caption_pil_image(test_pil[img], encoder, decoder, vocab, device))
        else:
            hyp = simple_tokenize(caption_image(f"{args.images_dir}/{img}", encoder, decoder, vocab, device))
        b1 = sentence_bleu(refs, hyp, weights=(1,0,0,0), smoothing_function=smooth)
        b4 = sentence_bleu(refs, hyp, weights=(.25,.25,.25,.25), smoothing_function=smooth)
        m  = meteor_score(refs, hyp)
        all_refs.append(refs)
        all_hyps.append(hyp)
        all_meteors.append(m)
        print(f"{img:<35} {b1:>7.3f} {b4:>7.3f} {m:>7.3f}  {' '.join(hyp)}")
        if bleu_table is not None and len(bleu_table.data) < TABLE_LIMIT:
            ref_str = " | ".join([" ".join(r) for r in refs])
            if args.flickr30k_hf:
                bleu_table.add_data(str(img), " ".join(hyp), ref_str, round(b1, 3), round(b4, 3), round(m, 3))
            else:
                bleu_table.add_data(wandb.Image(str(images_dir_abs / img)), " ".join(hyp), ref_str, round(b1, 3), round(b4, 3), round(m, 3))

    cb1 = corpus_bleu(all_refs, all_hyps, weights=(1,0,0,0))
    cb4 = corpus_bleu(all_refs, all_hyps, weights=(.25,.25,.25,.25))
    cm  = float(np.mean(all_meteors))
    print("-" * 110)
    print(f"[bleu] Corpus BLEU-1: {cb1:.3f}  BLEU-4: {cb4:.3f}  METEOR: {cm:.3f}")

    if use_wandb:
        wandb.log({"bleu/corpus_bleu1": cb1, "bleu/corpus_bleu4": cb4, "bleu/meteor": cm, "bleu/eval_table": bleu_table})
    # ------------------------------------------------

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
