"""Training script for the Flickr baseline image captioning model."""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from src.baseline.model import DecoderRNN, EncoderCNN
from src.baseline.sample import caption_image, caption_pil_image
from src.shared.caption_metrics import (
    aggregate_caption_scores,
    empty_caption_metrics,
    quality_breakdown_rows,
    score_caption_prediction,
)
from src.shared.dataset import get_loaders, get_loaders_hf, load_captions_df
from src.shared.losses import SemanticCrossEntropyLoss, build_soft_labels
from src.shared.vocabulary import (
    Vocabulary,
    build_vocab,
    build_vocab_hf,
    load_glove_weights,
    load_word2vec_weights,
    simple_tokenize,
)


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", default="dataset/Images")
    parser.add_argument("--captions-csv", default="dataset/captions.txt")
    parser.add_argument("--vocab-path", default="dataset/vocab.pkl")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--vocab-threshold", type=int, default=5)

    parser.add_argument("--embed-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--backbone", default="resnet152", choices=["resnet50", "resnet152", "efficientnet_b0", "vgg16"])
    parser.add_argument("--no-pretrained-backbone", action="store_true")
    parser.add_argument("--decoder-direction", default="uni", choices=["uni", "bidir"])

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=999)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-step", type=int, default=20)
    parser.add_argument("--scheduler", default="plateau", choices=["plateau", "cyclic"])
    parser.add_argument("--base-lr", type=float, default=1e-4)
    parser.add_argument("--max-lr", type=float, default=1e-3)
    parser.add_argument("--step-size-up-epochs", type=int, default=4)

    parser.add_argument("--glove-path", default=None)
    parser.add_argument("--word2vec-path", default=None)
    parser.add_argument("--word2vec-binary", action="store_true")
    parser.add_argument("--freeze-embeddings", action="store_true")
    parser.add_argument("--no-semantic-loss", action="store_true")
    parser.add_argument("--semantic-temp", type=float, default=10.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)

    parser.add_argument("--flickr30k-hf", action="store_true")
    parser.add_argument("--flickr30k-hf-cache", default="dataset/flickr30k_hf")
    parser.add_argument("--max-train-images", type=int, default=None)
    parser.add_argument("--max-val-images", type=int, default=None)
    parser.add_argument("--max-test-images", type=int, default=None)

    parser.add_argument("--skip-test-captioning", action="store_true")
    parser.add_argument("--checkpoint-mode", default="full", choices=["full", "light", "none"])
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--skip-local-artifacts", action="store_true")

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="altre_net")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-job-type", default=None)
    parser.add_argument("--wandb-tags", default="")
    parser.add_argument("--wandb-rich", action="store_true")
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def safe_save(obj, path, retries: int = 5):
    import os
    import shutil
    import tempfile
    import time as time_module

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(os.environ.get("CHECKPOINT_TMPDIR", "/tmp"))
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for attempt in range(retries):
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(prefix=f"{path.name}.", suffix=".tmp", dir=tmp_dir, delete=False) as f:
                tmp = Path(f.name)
            torch.save(obj, tmp, _use_new_zipfile_serialization=False)
            shutil.copy2(tmp, path)
            tmp.unlink(missing_ok=True)
            return
        except (RuntimeError, OSError, IOError) as exc:
            if tmp is not None:
                tmp.unlink(missing_ok=True)
            if attempt < retries - 1:
                print(f"[ckpt] save error ({exc}); retry {attempt + 1}/{retries}...")
                time_module.sleep(3)
            else:
                raise


def encoder_checkpoint_state(encoder, checkpoint_mode: str) -> dict:
    if checkpoint_mode == "full":
        return encoder.state_dict()
    if checkpoint_mode == "light":
        return {k: v for k, v in encoder.state_dict().items() if not k.startswith("net.")}
    return {}


def get_or_build_vocab(args) -> Vocabulary:
    vocab_path = Path(args.vocab_path)
    if vocab_path.exists():
        with open(vocab_path, "rb") as f:
            return pickle.load(f)
    vocab = build_vocab(args.captions_csv, threshold=args.vocab_threshold)
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    print(f"[vocab] built and saved to {vocab_path} (size={len(vocab)})")
    return vocab


@torch.no_grad()
def evaluate(encoder, decoder, loader, criterion, device) -> float:
    encoder.eval()
    decoder.eval()
    losses = []
    for images, captions, lengths in loader:
        images = images.to(device)
        captions = captions.to(device)
        targets = pack_padded_sequence(captions, lengths, batch_first=True).data
        features = encoder(images)
        outputs = decoder(features, captions, lengths)
        loss = criterion(outputs, targets)
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def evaluate_caption_metrics(encoder, decoder, vocab, image_ids, df_caps, pil_lookup, args, device) -> tuple[dict, dict]:
    encoder.eval()
    decoder.eval()
    samples = []
    for image_id in image_ids:
        refs = [simple_tokenize(c) for c in df_caps[df_caps["image"] == image_id]["caption"].tolist()]
        if not refs:
            continue
        if args.flickr30k_hf:
            hyp = simple_tokenize(caption_pil_image(pil_lookup[image_id], encoder, decoder, vocab, device))
        else:
            hyp = simple_tokenize(caption_image(f"{args.images_dir}/{image_id}", encoder, decoder, vocab, device))
        sample = {"image_id": image_id, "refs": refs, "hyp": hyp}
        sample.update(score_caption_prediction(refs, hyp))
        samples.append(sample)

    return aggregate_caption_scores(samples, prefix="val")


def build_scheduler(optimizer, args, steps_per_epoch: int):
    if args.scheduler == "cyclic":
        step_size_up = max(1, args.step_size_up_epochs * steps_per_epoch)
        return torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=args.base_lr,
            max_lr=args.max_lr,
            step_size_up=step_size_up,
            mode="triangular2",
            cycle_momentum=False,
        )
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)


def generation_supported(args) -> bool:
    return args.decoder_direction == "uni"


def maybe_caption_metrics(encoder, decoder, vocab, val_ids, df_caps, val_pil, args, device):
    if generation_supported(args):
        return evaluate_caption_metrics(encoder, decoder, vocab, val_ids, df_caps, val_pil, args, device)
    return empty_caption_metrics("val")


def main():
    args = parse_args()
    device = detect_device()
    print(f"[device] {device}")

    if args.checkpoint_mode != "none" or not args.skip_local_artifacts:
        Path(args.checkpoints_dir).mkdir(parents=True, exist_ok=True)

    if args.flickr30k_hf:
        from datasets import load_dataset

        print("[data] carregant Flickr30k HuggingFace...")
        hf_ds = load_dataset("nlphuji/flickr30k", trust_remote_code=True, cache_dir=args.flickr30k_hf_cache)
        vocab_path = Path(args.vocab_path)
        if vocab_path.exists():
            with open(vocab_path, "rb") as f:
                vocab = pickle.load(f)
            print(f"[vocab] carregat de {vocab_path} (size={len(vocab)})")
        else:
            print("[vocab] construint des de HF dataset...")
            vocab = build_vocab_hf(hf_ds, threshold=args.vocab_threshold)
            vocab_path.parent.mkdir(parents=True, exist_ok=True)
            with open(vocab_path, "wb") as f:
                pickle.dump(vocab, f)
            print(f"[vocab] built and saved to {vocab_path} (size={len(vocab)})")

        train_loader, val_loader, _ = get_loaders_hf(
            hf_ds,
            vocab,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_train_images=args.max_train_images,
            max_val_images=args.max_val_images,
            max_test_images=args.max_test_images,
        )
        full = hf_ds["test"]
        val_rows = full.filter(lambda x: x["split"] == "val")
        test_rows = full.filter(lambda x: x["split"] == "test")
        if args.max_val_images is not None:
            val_rows = val_rows.select(range(min(args.max_val_images, len(val_rows))))
        if args.max_test_images is not None:
            test_rows = test_rows.select(range(min(args.max_test_images, len(test_rows))))
        val_ids = [row["filename"] for row in val_rows]
        test_ids = [row["filename"] for row in test_rows]
        records = []
        for row in full:
            for cap in row["caption"]:
                records.append({"image": row["filename"], "caption": cap})
        import pandas as pd

        df_caps = pd.DataFrame(records)
        val_pil = {row["filename"]: row["image"] for row in val_rows}
        test_pil = {row["filename"]: row["image"] for row in test_rows}
    else:
        vocab = get_or_build_vocab(args)
        train_loader, val_loader, _, (_, val_ids, test_ids) = get_loaders(
            images_dir=args.images_dir,
            captions_csv=args.captions_csv,
            vocab=vocab,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_train_images=args.max_train_images,
            max_val_images=args.max_val_images,
            max_test_images=args.max_test_images,
        )
        df_caps = load_captions_df(args.captions_csv)
        val_pil = None
        test_pil = None

    print(f"[vocab] size = {len(vocab)}")
    print(f"[data] train batches={len(train_loader)}  val batches={len(val_loader)}")

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

    encoder = EncoderCNN(
        args.embed_size,
        backbone=args.backbone,
        pretrained=not args.no_pretrained_backbone,
    ).to(device)
    decoder = DecoderRNN(
        args.embed_size,
        args.hidden_size,
        len(vocab),
        args.num_layers,
        dropout=args.dropout,
        pretrained_weights=pretrained_weights,
        freeze_embeddings=args.freeze_embeddings,
        decoder_direction=args.decoder_direction,
    ).to(device)

    use_semantic_loss = pretrained_weights is not None and not args.no_semantic_loss
    if use_semantic_loss:
        soft_labels = build_soft_labels(decoder.embed.weight.data.cpu(), temperature=args.semantic_temp)
        criterion = SemanticCrossEntropyLoss(soft_labels).to(device)
        loss_tag = f"semantic-temp{args.semantic_temp}"
        print(f"[loss] SemanticCrossEntropy (temp={args.semantic_temp}) — soft labels des de {emb_type}")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        loss_tag = "ce" if args.label_smoothing == 0 else f"ce-ls{args.label_smoothing}"
        suffix = f" (label_smoothing={args.label_smoothing})" if args.label_smoothing > 0 else ""
        print(f"[loss] CrossEntropyLoss{suffix}")

    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer_lr = args.base_lr if args.scheduler == "cyclic" else args.lr
    optimizer = torch.optim.Adam(params, lr=optimizer_lr)
    scheduler = build_scheduler(optimizer, args, len(train_loader))

    use_wandb = args.wandb
    if use_wandb:
        import wandb

        wandb_tags = [tag for tag in args.wandb_tags.split(",") if tag]
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            group=args.wandb_group,
            job_type=args.wandb_job_type,
            tags=wandb_tags,
            config=vars(args),
        )
        wandb.config.update(
            {
                "vocab_size": len(vocab),
                "embedding_type": emb_type,
                "loss_type": loss_tag,
                "generation_supported": generation_supported(args),
            }
        )

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        decoder.train()
        t0 = time.time()
        epoch_losses = []

        for i, (images, captions, lengths) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            captions = captions.to(device, non_blocking=True)
            targets = pack_padded_sequence(captions, lengths, batch_first=True).data

            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            optimizer.step()
            if args.scheduler == "cyclic":
                scheduler.step()

            epoch_losses.append(loss.item())
            train_losses.append(loss.item())
            if i % args.log_step == 0:
                ppl = float(np.exp(min(loss.item(), 20)))
                print(
                    f"epoch {epoch}/{args.epochs}  step {i}/{len(train_loader)}  "
                    f"loss={loss.item():.4f}  ppl={ppl:.2f}  lr={optimizer.param_groups[0]['lr']:.2e}"
                )

        train_loss_epoch = float(np.mean(epoch_losses))
        train_ppl_epoch = float(np.exp(min(train_loss_epoch, 20)))
        val_loss = evaluate(encoder, decoder, val_loader, criterion, device)
        val_losses.append(val_loss)
        if args.scheduler == "plateau":
            scheduler.step(val_loss)
        val_ppl = float(np.exp(min(val_loss, 20)))
        elapsed = time.time() - t0
        caption_metrics, val_quality_summary = maybe_caption_metrics(encoder, decoder, vocab, val_ids, df_caps, val_pil, args, device)

        if generation_supported(args):
            print(
                f"== epoch {epoch} done  train_loss={train_loss_epoch:.4f}  val_loss={val_loss:.4f}  "
                f"bleu4={caption_metrics['val/bleu4']:.3f}  meteor={caption_metrics['val/meteor']:.3f}  "
                f"acc={caption_metrics['val/caption_accuracy_pct']:.1f}%  ({elapsed:.0f}s)"
            )
        else:
            print(
                f"== epoch {epoch} done  train_loss={train_loss_epoch:.4f}  val_loss={val_loss:.4f}  "
                f"caption-metrics=skipped (decoder_direction={args.decoder_direction})  ({elapsed:.0f}s)"
            )

        if use_wandb:
            log_payload = {
                "train/loss": train_loss_epoch,
                "train/perplexity": train_ppl_epoch,
                "val/loss": val_loss,
                "val/perplexity": val_ppl,
                **caption_metrics,
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
            }
            if args.wandb_rich and generation_supported(args) and val_quality_summary["total"] > 0:
                quality_table = wandb.Table(
                    data=quality_breakdown_rows(val_quality_summary),
                    columns=["quality_band", "count", "percentage"],
                )
                log_payload["val/caption_quality_bar"] = wandb.plot.bar(
                    quality_table,
                    "quality_band",
                    "percentage",
                    title="Validation Caption Quality (%)",
                )
            wandb.log(log_payload)

        if args.checkpoint_mode == "none":
            print("[ckpt] skipped (--checkpoint-mode none)")
        else:
            ckpt = {
                "epoch": epoch,
                "encoder": encoder_checkpoint_state(encoder, args.checkpoint_mode),
                "decoder": decoder.state_dict(),
                "vocab_size": len(vocab),
                "args": vars(args),
                "checkpoint_mode": args.checkpoint_mode,
            }
            if args.save_every_epoch:
                out = Path(args.checkpoints_dir) / f"ckpt_epoch{epoch}.pt"
                safe_save(ckpt, out)
                print(f"[ckpt] saved {out} ({args.checkpoint_mode})")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_out = Path(args.checkpoints_dir) / "ckpt_best.pt"
                safe_save(ckpt, best_out)
                print(f"[best] new best val_loss={best_val_loss:.4f} -> saved ckpt_best.pt")

    steps_per_epoch = len(train_loader)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(train_losses, alpha=0.6, label="train (per batch)")
    for epoch in range(1, args.epochs + 1):
        axes[0].axvline(epoch * steps_per_epoch, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_xlabel("batch")
    axes[0].set_ylabel("cross-entropy loss")
    axes[0].set_title("Train loss")
    axes[0].legend()

    axes[1].plot(range(1, len(val_losses) + 1), val_losses, marker="o", label="val")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("cross-entropy loss")
    axes[1].set_title("Val loss per epoch")
    axes[1].legend()

    if args.skip_local_artifacts:
        plt.close()
        print("[plot] skipped (--skip-local-artifacts)")
    else:
        plt.tight_layout()
        plot_path = Path(args.checkpoints_dir) / "loss_curve.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"[plot] saved {plot_path}")

    final_test_metrics = {}
    if args.skip_test_captioning or not generation_supported(args):
        reason = "requested via --skip-test-captioning" if args.skip_test_captioning else f"decoder_direction={args.decoder_direction}"
        print(f"[test] caption generation skipped ({reason})")
        if use_wandb:
            wandb.summary["test_captioning_skipped"] = True
            wandb.summary["test_captioning_skip_reason"] = reason
    elif args.checkpoint_mode == "none":
        print("[test] caption generation skipped (--checkpoint-mode none leaves no best checkpoint)")
    else:
        import nltk

        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
        print("\n[bleu+meteor] avaluant sobre el conjunt de test...")
        best_ckpt = torch.load(Path(args.checkpoints_dir) / "ckpt_best.pt", map_location=device)
        encoder.load_state_dict(best_ckpt["encoder"], strict=best_ckpt.get("checkpoint_mode", "full") == "full")
        decoder.load_state_dict(best_ckpt["decoder"])
        encoder.eval()
        decoder.eval()

        eval_samples = []
        bleu_table = (
            wandb.Table(
                columns=[
                    "image",
                    "generated_caption",
                    "reference_captions",
                    "BLEU-1",
                    "BLEU-4",
                    "METEOR",
                    "token_overlap_f1",
                    "quality_band",
                ]
            )
            if use_wandb and args.wandb_rich
            else None
        )
        images_dir_abs = Path(args.images_dir).resolve()
        table_limit = 200

        print(f"Evaluating {len(test_ids)} test images...")
        print(f"{'Image':<35} {'BLEU-1':>7} {'BLEU-4':>7} {'METEOR':>7} {'QUAL':>14}  Caption")
        print("-" * 130)
        for image_id in test_ids:
            refs = [simple_tokenize(c) for c in df_caps[df_caps["image"] == image_id]["caption"].tolist()]
            if args.flickr30k_hf:
                hyp = simple_tokenize(caption_pil_image(test_pil[image_id], encoder, decoder, vocab, device))
            else:
                hyp = simple_tokenize(caption_image(f"{args.images_dir}/{image_id}", encoder, decoder, vocab, device))
            sample = {"image_id": image_id, "refs": refs, "hyp": hyp}
            sample.update(score_caption_prediction(refs, hyp))
            eval_samples.append(sample)
            print(
                f"{image_id:<35} {sample['bleu1']:>7.3f} {sample['bleu4']:>7.3f} {sample['meteor']:>7.3f} "
                f"{sample['quality_display']:>14}  {' '.join(hyp)}"
            )
            if bleu_table is not None and len(bleu_table.data) < table_limit:
                ref_str = " | ".join([" ".join(ref) for ref in refs])
                if args.flickr30k_hf:
                    bleu_table.add_data(
                        str(image_id),
                        " ".join(hyp),
                        ref_str,
                        round(sample["bleu1"], 3),
                        round(sample["bleu4"], 3),
                        round(sample["meteor"], 3),
                        round(sample["overlap_f1"], 3),
                        sample["quality_display"],
                    )
                else:
                    from PIL import Image as PILImage

                    pil_img = PILImage.open(str(images_dir_abs / image_id)).convert("RGB")
                    bleu_table.add_data(
                        wandb.Image(pil_img),
                        " ".join(hyp),
                        ref_str,
                        round(sample["bleu1"], 3),
                        round(sample["bleu4"], 3),
                        round(sample["meteor"], 3),
                        round(sample["overlap_f1"], 3),
                        sample["quality_display"],
                    )

        test_caption_metrics, test_quality_summary = aggregate_caption_scores(eval_samples, prefix="test")
        final_test_metrics = test_caption_metrics
        print("-" * 130)
        print(
            f"[caption] Corpus BLEU-1: {test_caption_metrics['test/bleu1']:.3f}  "
            f"BLEU-4: {test_caption_metrics['test/bleu4']:.3f}  "
            f"METEOR: {test_caption_metrics['test/meteor']:.3f}  "
            f"ACC: {test_caption_metrics['test/caption_accuracy_pct']:.1f}%"
        )

        if use_wandb:
            caption_payload = {
                "caption/corpus_bleu1": test_caption_metrics["test/bleu1"],
                "caption/corpus_bleu4": test_caption_metrics["test/bleu4"],
                "caption/meteor": test_caption_metrics["test/meteor"],
                "caption/token_overlap_f1": test_caption_metrics["test/token_overlap_f1"],
                "caption/accuracy_pct": test_caption_metrics["test/caption_accuracy_pct"],
                "caption/partial_pct": test_caption_metrics["test/caption_partial_pct"],
                "caption/bad_pct": test_caption_metrics["test/caption_bad_pct"],
                "caption/quality_score_pct": test_caption_metrics["test/caption_quality_score_pct"],
            }
            if args.wandb_rich:
                quality_table = wandb.Table(
                    data=quality_breakdown_rows(test_quality_summary),
                    columns=["quality_band", "count", "percentage"],
                )
                caption_payload["caption/eval_table"] = bleu_table
                caption_payload["caption/quality_bar"] = wandb.plot.bar(
                    quality_table,
                    "quality_band",
                    "percentage",
                    title="Test Caption Quality (%)",
                )
            wandb.log(caption_payload)

    metrics_summary = {
        "run_name": args.run_name,
        "architecture": "baseline",
        "wandb_project": args.wandb_project,
        "wandb_group": args.wandb_group,
        "backbone": args.backbone,
        "embedding_type": emb_type,
        "loss_type": loss_tag,
        "decoder_direction": args.decoder_direction,
        "best_val_loss": None if best_val_loss == float("inf") else best_val_loss,
        "final_train_loss": None if not train_losses else train_losses[-1],
        "final_val_loss": None if not val_losses else val_losses[-1],
        "test_metrics": final_test_metrics,
    }
    if args.skip_local_artifacts:
        print("[metrics] local metrics skipped (--skip-local-artifacts)")
    else:
        metrics_path = Path(args.checkpoints_dir) / "metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics_summary, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[metrics] saved {metrics_path}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
