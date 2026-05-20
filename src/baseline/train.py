"""Training script for the Flickr8k Image Captioning baseline.

Usage:
    python -m src.train --epochs 5 --batch-size 32 --wandb
"""
from __future__ import annotations

import argparse
import os
import pickle
import time
from pathlib import Path

import pandas as pd  # per llegir el CSV de captions durant l'avaluació BLEU
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn # aquí s'utilitza per crear la loss nn.CrossEntropyLoss()
from torch.nn.utils.rnn import pack_padded_sequence # per treballar amb seqüències de longitud variable.
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction  # mètriques BLEU
from nltk.translate.meteor_score import meteor_score  # mètrica METEOR (té en compte sinònims)

from src.shared.dataset import get_loaders, get_loaders_hf, get_loaders_coco, split_image_ids, load_captions_df, COCODataset
from src.shared.losses import SemanticCrossEntropyLoss, TopKSemanticLoss, build_glove_similarity

from src.baseline.model import DecoderRNN, EncoderCNN # les dues xarxes principals.
from src.baseline.sample import caption_image, caption_pil_image  # per generar captions durant l'avaluació BLEU
from src.shared.vocabulary import Vocabulary, build_vocab, build_vocab_hf, simple_tokenize, load_glove_weights, load_word2vec_weights, load_fasttext_weights


def parse_args(): # a llegir tots els arguments
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", default="dataset/Images") # directori imatges
    p.add_argument("--captions-csv", default="dataset/captions.txt") # path fitxer captions
    p.add_argument("--vocab-path", default="dataset/vocab.pkl") # path on es guarda o carrega el vocab
    p.add_argument("--checkpoints-dir", default="checkpoints") # directori on es guarden els checkpoints de l'entrenament (epoch, vocab_size, args...) i models entrenats
    p.add_argument("--vocab-threshold", type=int, default=5) # mínim de vegades ha d'aparèixer una paraula per entrar al vocabulari (default 5, si no entra --> <unk>)

    p.add_argument("--embed-size", type=int, default=256) # mida dels embeddings de paraules (sobreescrit per GloVe/W2V/FastText)
    p.add_argument("--encoder-size", type=int, default=None,
                   help="Mida del vector de la imatge (sortida de EncoderCNN). Si no s'especifica, usa embed-size.")
    p.add_argument("--hidden-size", type=int, default=512) # mida de l'estat ocult de la LSTM
    p.add_argument("--num-layers", type=int, default=1) # nombre de capes apilades de la LSTM (profunditat)
    p.add_argument("--dropout", type=float, default=0.5) # probabilitat de dropout a la LSTM (regularització que apaga neurones aleatòriament durant l'entrenament per evitar overfitting)
    p.add_argument("--scheduled-sampling", action="store_true", help="Activa scheduled sampling (decay de teacher forcing)")
    p.add_argument("--ss-start",  type=float, default=1.0, help="Probabilitat inicial de teacher forcing (default 1.0)")
    p.add_argument("--ss-end",    type=float, default=0.3, help="Probabilitat mínima de teacher forcing (default 0.3)")
    p.add_argument("--ss-epochs", type=int,   default=None, help="Epochs fins arribar a ss-end. Després es queda fix. Si None, usa totes les epochs.")
    p.add_argument("--backbone", default="resnet152") # quina CNN preentrenada utilitzar com a encoder (resnet50 o resnet152)

    p.add_argument("--epochs", type=int, default=20) # numero de passades completes del train
    p.add_argument("--patience", type=int, default=999) # nombre d'epochs que esperarem sense millora en la val_loss abans de parar l'entrenament (early stopping)
    p.add_argument("--batch-size", type=int, default=32) # quantes mostres entrenen el model a cada pas 
    p.add_argument("--num-workers", type=int, default=2) # quants processos paral·lels carregaran dades
    p.add_argument("--lr", type=float, default=1e-3) # de l'optimitzador Adam (la mida del pas d'actualització dels pesos)
    p.add_argument("--log-step", type=int, default=20) # cada quants batches s'imprimeixen mètriques (loss, perplexity)

    p.add_argument("--glove-path", default=None,
                   help="Ruta al fitxer GloVe per inicialitzar embeddings.")
    p.add_argument("--word2vec-path", default=None,
                   help="Ruta al fitxer Word2Vec (.bin o .txt). S'ignora si --glove-path s'especifica.")
    p.add_argument("--word2vec-binary", action="store_true",
                   help="Indica que el fitxer Word2Vec és en format binari (.bin).")
    p.add_argument("--fasttext-path", default=None,
                   help="Ruta al fitxer FastText (.vec o .txt). S'ignora si --glove-path o --word2vec-path s'especifiquen.")
    p.add_argument("--freeze-embeddings", action="store_true",
                   help="Si s'activa, els pesos (GloVe o Word2Vec) no s'actualitzen durant l'entrenament.")
    p.add_argument("--fine-tune-encoder", action="store_true",
                   help="Desglacen layer4 del ResNet per fine-tuning. Millora les features però és més lent.")
    p.add_argument("--label-smoothing", type=float, default=0.0,
                   help="Label smoothing per CrossEntropyLoss (0.0 = desactivat, 0.1 recomanat).")
    p.add_argument("--semantic-loss", action="store_true",
                   help="Usa TopKSemanticLoss (top-10 GloVe) en lloc de CrossEntropyLoss. Requereix --glove-path.")
    p.add_argument("--semantic-loss-k", type=int, default=10,
                   help="K per TopKSemanticLoss (default 10).")
    p.add_argument("--semantic-loss-alpha", type=float, default=0.2,
                   help="Pes de les paraules similars a TopKSemanticLoss (default 0.2).")

    p.add_argument("--scheduler", default="plateau",
                   choices=["none", "plateau", "step", "cosine", "cyclic", "cyclic2",
                            "onecycle", "polynomial", "cosine_warm"],
                   help="LR scheduler a utilitzar.")
    p.add_argument("--scheduler-step-size", type=int, default=5,
                   help="Step size per StepLR (cada quantes epochs es redueix el LR).")
    p.add_argument("--scheduler-gamma", type=float, default=0.5,
                   help="Factor de reducció per StepLR.")
    p.add_argument("--scheduler-t0", type=int, default=5,
                   help="T_0 per CosineAnnealingWarmRestarts (epochs del primer cicle).")

    # ── COCO 2017 ─────────────────────────────────────────────────────────
    p.add_argument("--coco", action="store_true",
                   help="Usa el dataset COCO 2017 en lloc de Flickr8k.")
    p.add_argument("--coco-dir", default="/home/datasets/coco",
                   help="Directori arrel de COCO (amb train2017/, val2017/ i annotations/).")
    p.add_argument("--coco-max-images", type=int, default=10000,
                   help="Nombre màxim d'imatges de train de COCO a usar (default 10000).")

    # ── Flickr30k HuggingFace ──────────────────────────────────────────────
    p.add_argument("--flickr30k-hf", action="store_true",
                   help="Usa el dataset Flickr30k de HuggingFace (nlphuji/flickr30k) en lloc del CSV.")
    p.add_argument("--flickr30k-hf-cache", default="dataset/flickr30k_hf",
                   help="Carpeta cache del dataset HuggingFace.")

    p.add_argument("--wandb", action="store_true") # argument que activa wandb
    p.add_argument("--wandb-project", default="image-captioning") # nom del projecte a wandb
    p.add_argument("--wandb-entity", default=None) # nom de l'entitat (usuari o organització) a wandb, si es deixa None s'utilitzarà l'entitat per defecte de l'usuari
    p.add_argument("--run-name", default=None) # nom l'execució concreta, si es deixa None s'utilitzarà un nom generat automàticament basat en la data i hora actual
    return p.parse_args()


def safe_save(obj, path, retries: int = 5):
    """torch.save escrivint primer a /tmp (local) i després copiant al NFS."""
    import shutil, time
    path = Path(path)
    tmp_local = Path("/tmp") / f".tmp_{os.getpid()}_{path.name}"
    for attempt in range(retries):
        try:
            torch.save(obj, tmp_local)
            shutil.copy2(str(tmp_local), str(path))
            tmp_local.unlink(missing_ok=True)
            return
        except RuntimeError:
            if attempt < retries - 1:
                print(f"[ckpt] error NFS (intent {attempt+1}/{retries}), reintentant...")
                time.sleep(3)
            else:
                tmp_local.unlink(missing_ok=True)
                raise


def build_scheduler(optimizer, args, steps_per_epoch: int):
    s = args.scheduler
    if s == "none":
        return None
    if s == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=2, factor=0.5)
    if s == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)
    if s == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs)
    if s == "cyclic":
        return torch.optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=args.lr / 10, max_lr=args.lr,
            step_size_up=steps_per_epoch * 2, mode="triangular", cycle_momentum=False)
    if s == "cyclic2":
        return torch.optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=args.lr / 10, max_lr=args.lr,
            step_size_up=steps_per_epoch * 2, mode="triangular2", cycle_momentum=False)
    if s == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr,
            steps_per_epoch=steps_per_epoch, epochs=args.epochs)
    if s == "polynomial":
        return torch.optim.lr_scheduler.PolynomialLR(
            optimizer, total_iters=args.epochs, power=1.0)
    if s == "cosine_warm":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.scheduler_t0)
    raise ValueError(f"Scheduler desconegut: {s}")


_PER_BATCH_SCHEDULERS = {"cyclic", "cyclic2", "onecycle"}


def get_or_build_vocab(args) -> Vocabulary: # funció que retorna l'objecte Vocabulary
    vp = Path(args.vocab_path) # ruta del vocab
    if vp.exists(): # si ja existeix el fitxer del vocab, el carrega i el retorna
        with open(vp, "rb") as f:
            return pickle.load(f)
    vocab = build_vocab(args.captions_csv, threshold=args.vocab_threshold) # si no existeix, el construeix a partir de les captions i threshold
    vp.parent.mkdir(parents=True, exist_ok=True)
    with open(vp, "wb") as f:
        pickle.dump(vocab, f) # el guarda 
    print(f"[vocab] built and saved to {vp} (size={len(vocab)})")
    return vocab # i el retorna


@torch.no_grad()
def evaluate_autoregressive(encoder, decoder, loader, criterion, device) -> float:
    """Val loss de cross-entropy en mode free-running: el model usa les seves pròpies prediccions,
    no veu les captions reals. Reflecteix les condicions reals d'inferència."""
    encoder.eval()
    decoder.eval()
    losses = []
    for images, captions, lengths in loader:
        images = images.to(device, non_blocking=True)
        captions = captions.to(device, non_blocking=True)
        features = encoder(images)
        # p_teacher=0.0 → sempre usa la predicció del model, mai les captions reals
        outputs = decoder.forward_scheduled(features, captions, p_teacher=0.0)  # [B, T-1, vocab_size]
        targets = captions[:, 1:]  # [B, T-1]
        mask = targets != 0        # ignora padding
        loss = criterion(outputs[mask], targets[mask])
        losses.append(loss.item())
    return float(np.mean(losses))


class _NoWordNet:
    def synsets(self, word, pos=None): return []
    def morphy(self, word, pos=None):  return None

_no_wn = _NoWordNet()

@torch.no_grad()
def evaluate_bleu(encoder, decoder, vocab, val_ids, df_caps, val_pil, args, device) -> dict:
    """Calcula BLEU-1, BLEU-4 i METEOR (amb i sense WordNet) sobre el val set."""
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
            hyp = simple_tokenize(caption_pil_image(val_pil[img], encoder, decoder, vocab, device))
        else:
            import os as _os
            img_path = img if _os.path.isabs(img) else f"{args.images_dir}/{img}"
            hyp = simple_tokenize(caption_image(img_path, encoder, decoder, vocab, device))
        all_refs.append(refs)
        all_hyps.append(hyp)
        all_meteors.append(meteor_score(refs, hyp))
    b1 = corpus_bleu(all_refs, all_hyps, weights=(1, 0, 0, 0))
    b4 = corpus_bleu(all_refs, all_hyps, weights=(.25, .25, .25, .25))
    m  = float(np.mean(all_meteors))

    return {"val/bleu1": b1, "val/bleu4": b4, "val/meteor": m}


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # entrena amb gpu si està disponible, si no cpu
    print(f"[device] {device}") # imprimeix quin s'utilitzarà

    Path(args.checkpoints_dir).mkdir(parents=True, exist_ok=True) # crea la carpeta de checkpoints si no existeix

    # ── Carrega dataset i vocab (CSV, HuggingFace o COCO) ─────────────────
    if args.coco:
        import json, pickle
        from src.shared.vocabulary import build_vocab as _build_vocab
        coco_dir = Path(args.coco_dir)
        vp = Path(args.vocab_path)
        if vp.exists():
            with open(vp, "rb") as f:
                vocab = pickle.load(f)
            print(f"[vocab] carregat de {vp} (size={len(vocab)})")
        else:
            print("[vocab] construint des de COCO...")
            with open(coco_dir / "annotations/captions_train2017.json") as f:
                coco_data = json.load(f)
            from collections import Counter
            counter: Counter = Counter()
            for ann in coco_data["annotations"]:
                counter.update(simple_tokenize(ann["caption"]))
            vocab = Vocabulary()
            for word, cnt in counter.items():
                if cnt >= args.vocab_threshold:
                    vocab.add_word(word)
            vp.parent.mkdir(parents=True, exist_ok=True)
            with open(vp, "wb") as f:
                pickle.dump(vocab, f)
            print(f"[vocab] size={len(vocab)} saved to {vp}")

        train_loader, val_loader = get_loaders_coco(
            coco_dir=coco_dir,
            vocab=vocab,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_train_images=args.coco_max_images,
        )
        # construeix df i ids del val de COCO per evaluate_bleu
        import json as _json, random as _random
        with open(coco_dir / "annotations/captions_val2017.json") as f:
            coco_val_data = _json.load(f)
        _id2file = {img["id"]: img["file_name"] for img in coco_val_data["images"]}
        _all_val_ids = list(_id2file.keys())
        _rng = _random.Random(42)
        _sel = set(_rng.sample(_all_val_ids, min(1000, len(_all_val_ids))))
        _rows = [(str(coco_dir / "val2017" / _id2file[a["image_id"]]), a["caption"])
                 for a in coco_val_data["annotations"] if a["image_id"] in _sel]
        import pandas as _pd2
        coco_val_df = _pd2.DataFrame(_rows, columns=["image", "caption"])
        coco_val_ids = coco_val_df["image"].unique().tolist()
        args.images_dir = str(coco_dir / "val2017")
        val_ids = None
        df_caps = None
    elif args.flickr30k_hf:
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

        full = hf_ds["test"]
        val_rows  = full.filter(lambda x: x["split"] == "val")
        test_rows = full.filter(lambda x: x["split"] == "test")
        val_ids   = [r["filename"] for r in val_rows]
        test_ids  = [r["filename"] for r in test_rows]
        records = []
        for r in full:
            for cap in r["caption"]:
                records.append({"image": r["filename"], "caption": cap})
        import pandas as _pd
        df_caps_hf = _pd.DataFrame(records)
        val_pil  = {r["filename"]: r["image"] for r in val_rows}
        test_pil = {r["filename"]: r["image"] for r in test_rows}
    else:
        vocab = get_or_build_vocab(args) # carrega o construeix el vocabulari
        train_loader, val_loader, _, (_, val_ids, _) = get_loaders(
            images_dir=args.images_dir,
            captions_csv=args.captions_csv,
            vocab=vocab,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        ) # crea els data loaders

    print(f"[vocab] size = {len(vocab)}") # mostra mida del vocabulari
    print(f"[data] train batches={len(train_loader)}  val batches={len(val_loader)}") # mira quants batches hi ha al train i a la validació

    # --- Embeddings: scratch / GloVe / Word2Vec (fine-tuned o frozen) ---
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
    elif args.fasttext_path:
        pretrained_weights, ft_dim = load_fasttext_weights(args.fasttext_path, vocab)
        pretrained_weights = pretrained_weights.to(device)
        args.embed_size = ft_dim
        emb_type = "fasttext-frozen" if args.freeze_embeddings else "fasttext-finetune"
    else:
        emb_type = "scratch"
    # encoder_size independent d'embed_size (si no s'especifica, usa embed_size)
    if args.encoder_size is None:
        args.encoder_size = args.embed_size
    print(f"[embeddings] tipus={emb_type}  embed_size={args.embed_size}  encoder_size={args.encoder_size}")

    encoder = EncoderCNN(args.encoder_size, backbone=args.backbone, fine_tune_encoder=args.fine_tune_encoder).to(device)
    decoder = DecoderRNN(args.encoder_size, args.embed_size, args.hidden_size, len(vocab), args.num_layers,
                         dropout=args.dropout, pretrained_weights=pretrained_weights,
                         freeze_embeddings=args.freeze_embeddings).to(device)

    if args.semantic_loss:
        if pretrained_weights is None:
            raise ValueError("--semantic-loss requereix --glove-path")
        soft_labels = build_glove_similarity(pretrained_weights.cpu()).to(device)
        criterion = TopKSemanticLoss(soft_labels, k=args.semantic_loss_k, alpha=args.semantic_loss_alpha).to(device)
        print(f"[loss] TopKSemanticLoss (k={args.semantic_loss_k}, alpha={args.semantic_loss_alpha})")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        ls_tag = f" (label_smoothing={args.label_smoothing})" if args.label_smoothing > 0 else ""
        print(f"[loss] CrossEntropyLoss{ls_tag}")
    # Quan fine_tune_encoder=True, layer4 s'entrena amb lr/10 per no destruir els pesos d'ImageNet
    # all_params s'usa per clip_grad_norm_ (necessita llista de tensors, no dicts)
    all_params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    if args.fine_tune_encoder:
        all_params += list(encoder.net.layer4.parameters())
        optimizer_params = [
            {"params": list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters()), "lr": args.lr},
            {"params": list(encoder.net.layer4.parameters()), "lr": args.lr / 10},
        ]
    else:
        optimizer_params = all_params
    optimizer = torch.optim.Adam(optimizer_params, lr=args.lr)
    scheduler = build_scheduler(optimizer, args, steps_per_epoch=len(train_loader))
    print(f"[scheduler] {args.scheduler}")

    use_wandb = args.wandb 
    if use_wandb: # si hem activat wandb
        import wandb
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.run_name, config=vars(args), dir="/tmp")
        wandb.config.update({"vocab_size": len(vocab), "embedding_type": emb_type}) # afegeix mida del vocabulari i tipus d'embedding a wandb

    train_losses: list[float] = [] # per guardar les losses de l'entrenament
    val_losses: list[float] = []   # CE loss de validació (free-running, sense teacher forcing)
    val_bleu4s: list[float] = []   # BLEU-4 de validació (greedy) per epoch

    best_val_bleu4 = 0.0 # inicialitza el millor BLEU-4 com 0
    global_step = 0 # comptador de batches processats
    for epoch in range(1, args.epochs + 1): # bucle d'epochs
        encoder.train() # posa l'encoder i el decoder en mode entrenament
        decoder.train()
        t0 = time.time() # guarda el temps inicial de la epoch
        epoch_losses = []

        # Scheduled sampling: decay lineal de p_teacher cada epoch
        if args.scheduled_sampling:
            ss_decay_epochs = args.ss_epochs if args.ss_epochs is not None else args.epochs
            decay = (args.ss_start - args.ss_end) / max(ss_decay_epochs - 1, 1)
            p_teacher = max(args.ss_end, args.ss_start - (epoch - 1) * decay)
        else:
            p_teacher = 1.0

        for i, (images, captions, lengths) in enumerate(train_loader): # recorre tots els batches d'entrenament; i --> index del batch, els altres són les dades del batch
            images = images.to(device, non_blocking=True) # mou les imatges a gpu o cpu. non_blocking pot accelerar la transferència si el DataLoader utilitza memòria pinned (que ho fa)
            captions = captions.to(device, non_blocking=True) # mou les captions al dispositiu
            dec_lengths = [l - 1 for l in lengths]

            features = encoder(images) # passa les imatges per la CNN

            if args.scheduled_sampling:
                outputs = decoder.forward_scheduled(features, captions, p_teacher)  # [B, T-1, vocab]
                targets = captions[:, 1:]                                            # [B, T-1]
                mask    = targets != 0
                loss    = criterion(outputs[mask], targets[mask])
            else:
                targets = pack_padded_sequence(captions[:, 1:], dec_lengths, batch_first=True).data
                outputs = decoder(features, captions[:, :-1], dec_lengths)
                loss    = criterion(outputs, targets)

            # BACKPROPAGATION
            optimizer.zero_grad() # posa els gradients a zero (perquè no s'acumulin amb els antics)
            loss.backward() # calcula els gradients
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=5.0)
            optimizer.step() # actualitza els pesos --> APRÈN yuppi
            if scheduler is not None and args.scheduler in _PER_BATCH_SCHEDULERS:
                scheduler.step()

            global_step += 1 # sumem 1 al comptador de batchos
            epoch_losses.append(loss.item())
            train_losses.append(loss.item()) # guarda la loss del batch
            if i % args.log_step == 0: # comprova si toca imprimir la informació
                ppl = float(np.exp(min(loss.item(), 20))) # calcula la perplexity (com més baixa millor)
                print(f"epoch {epoch}/{args.epochs}  step {i}/{len(train_loader)}  "
                      f"loss={loss.item():.4f}  ppl={ppl:.2f}") # imprimeix info de l'entrenament

        train_loss_epoch = float(np.mean(epoch_losses))
        train_ppl_epoch = float(np.exp(min(train_loss_epoch, 20)))

        # Val loss free-running: CE sense teacher forcing (el model usa les seves pròpies prediccions)
        val_loss = evaluate_autoregressive(encoder, decoder, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_ppl = float(np.exp(min(val_loss, 20)))

        elapsed = time.time() - t0 # es tanca el temps per saber quan ha durat la epoch

        # Validació: generació greedy sense teacher forcing → BLEU/METEOR reals
        if args.coco:
            _df_caps_eval = coco_val_df
            _val_pil_eval = None
            _val_ids_eval = coco_val_ids
        elif args.flickr30k_hf:
            _df_caps_eval = df_caps_hf
            _val_pil_eval = val_pil
            _val_ids_eval = val_ids
        else:
            _df_caps_eval = load_captions_df(args.captions_csv)
            _val_pil_eval = None
            _val_ids_eval = val_ids
        bleu_metrics = evaluate_bleu(encoder, decoder, vocab, _val_ids_eval, _df_caps_eval, _val_pil_eval, args, device)
        val_bleu4 = bleu_metrics["val/bleu4"]
        val_bleu4s.append(val_bleu4)

        if scheduler is not None and args.scheduler not in _PER_BATCH_SCHEDULERS:
            if args.scheduler == "plateau":
                scheduler.step(-val_bleu4)  # plateau minimitza, negatiu per maximitzar BLEU-4
            else:
                scheduler.step()

        print(f"== epoch {epoch} done  train_loss={train_loss_epoch:.4f}  val_loss(free)={val_loss:.4f}  "
              f"val/bleu4={val_bleu4:.3f}  val/meteor={bleu_metrics['val/meteor']:.3f}  ({elapsed:.0f}s)")
        if use_wandb:
            wandb.log({
                "train/loss": train_loss_epoch,
                "train/perplexity": train_ppl_epoch,
                "val/loss": val_loss,
                "val/perplexity": val_ppl,
                **bleu_metrics,
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                **({"train/p_teacher": p_teacher} if args.scheduled_sampling else {}),
            }) # registra metriques a wandb

        ckpt = { # diccionari amb la info que es vol guardar
            "epoch": epoch, # en quina epoch s'ha guardat el checkpoin
            "encoder": encoder.state_dict(), # guarda pesos de l'encoder
            "decoder": decoder.state_dict(), # del decoder
            "vocab_size": len(vocab),
            "args": vars(args),
        }
        if val_bleu4 > best_val_bleu4:
            best_val_bleu4 = val_bleu4
            safe_save(ckpt, Path(args.checkpoints_dir) / "ckpt_best.pt")
            print(f"[best] new best val/bleu4={best_val_bleu4:.4f} -> saved ckpt_best.pt")

    # --- loss curve ---
    steps_per_epoch = len(train_loader) # quarda quants batches té una epoch
    fig, axes = plt.subplots(1, 2, figsize=(12, 4)) # figura amb 2 gràfiques en una fila

    axes[0].plot(train_losses, alpha=0.6, label="train (per batch)") # la de le'squerra és la gràfica d'entrenament
    for e in range(1, args.epochs + 1):
        axes[0].axvline(e * steps_per_epoch, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_xlabel("batch")
    axes[0].set_ylabel("cross-entropy loss")
    axes[0].set_title("Train loss")
    axes[0].legend()

    ax2b = axes[1].twinx()
    axes[1].plot(range(1, len(val_losses) + 1), val_losses, marker="o", color="orange", label="val loss (free-running)")
    ax2b.plot(range(1, len(val_bleu4s) + 1), val_bleu4s, marker="s", color="green", label="val BLEU-4 (greedy)")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("CE loss (free-running)", color="orange")
    ax2b.set_ylabel("BLEU-4", color="green")
    axes[1].set_title("Validació sense teacher forcing")
    axes[1].legend(loc="upper left"); ax2b.legend(loc="upper right")

    plt.tight_layout()
    plot_path = Path(args.checkpoints_dir) / "loss_curve.png" # on es guarda la imatge
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[plot] saved {plot_path}")
    # ------------------

    # --- BLEU + METEOR evaluation on test set (millor checkpoint) ---
    import nltk
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    print("\n[bleu+meteor] avaluant sobre el conjunt de test...")
    best_ckpt = torch.load(Path(args.checkpoints_dir) / "ckpt_best.pt", map_location=device)
    encoder.load_state_dict(best_ckpt["encoder"])  # carrega pesos del millor model
    decoder.load_state_dict(best_ckpt["decoder"])
    encoder.eval()
    decoder.eval()

    if args.coco:
        print("[test] avaluant sobre val COCO (1000 imatges) per mètriques finals...")
        best_ckpt = torch.load(Path(args.checkpoints_dir) / "ckpt_best.pt", map_location=device)
        encoder.load_state_dict(best_ckpt["encoder"])
        decoder.load_state_dict(best_ckpt["decoder"])
        encoder.eval(); decoder.eval()

        # ── Greedy ───────────────────────────────────────────────────────
        print("[greedy] avaluant...")
        gm = evaluate_bleu(encoder, decoder, vocab, coco_val_ids, coco_val_df, None, args, device)
        cb1, cb4, cm = gm["val/bleu1"], gm["val/bleu4"], gm["val/meteor"]
        print(f"  Greedy  BLEU-1={cb1:.3f}  BLEU-4={cb4:.3f}  METEOR={cm:.3f}")

        # ── Beam search k=3 ──────────────────────────────────────────────
        from src.baseline.sample import caption_image_beam
        from nltk.translate.bleu_score import corpus_bleu as _cb
        from nltk.translate.meteor_score import meteor_score as _ms
        print("[beam k=3] avaluant...")
        b_refs, b_hyps, b_meteors = [], [], []
        for img in coco_val_ids:
            caps = coco_val_df[coco_val_df["image"] == img]["caption"].tolist()
            if not caps: continue
            try:
                hyp = simple_tokenize(caption_image_beam(img, encoder, decoder, vocab, device, beam_size=3))
            except Exception: continue
            refs = [simple_tokenize(c) for c in caps]
            b_refs.append(refs); b_hyps.append(hyp)
            b_meteors.append(_ms(refs, hyp))
        bb1 = _cb(b_refs, b_hyps, weights=(1,0,0,0))
        bb4 = _cb(b_refs, b_hyps, weights=(.25,.25,.25,.25))
        bm  = float(np.mean(b_meteors))
        print(f"  Beam k=3  BLEU-1={bb1:.3f}  BLEU-4={bb4:.3f}  METEOR={bm:.3f}")

        if use_wandb:
            wandb.log({
                "bleu/corpus_bleu1":      cb1,  "bleu/corpus_bleu4":      cb4,  "bleu/meteor":      cm,
                "bleu/beam3_bleu1":       bb1,  "bleu/beam3_bleu4":       bb4,  "bleu/beam3_meteor": bm,
            })
            wandb.finish()
        return
    if args.flickr30k_hf:
        df_caps = df_caps_hf
    else:
        _, _, test_ids = split_image_ids(args.captions_csv)  # agafa els IDs del test set
        df_caps = load_captions_df(args.captions_csv)  # llegeix totes les captions (Flickr8k o Flickr30k)
    smooth = SmoothingFunction().method1  # suavitzat per evitar BLEU-4 = 0

    all_refs, all_hyps = [], []
    all_meteors = []
    bleu_table = wandb.Table(columns=["image", "generated_caption", "reference_captions", "BLEU-1", "BLEU-4", "METEOR"]) if use_wandb else None
    images_dir_abs = Path(args.images_dir).resolve()
    TABLE_LIMIT = 200  # WandB no renderitza bé les imatges amb >200 files

    print(f"Evaluating {len(test_ids)} test images...")
    print(f"{'Image':<35} {'BLEU-1':>7} {'BLEU-4':>7} {'METEOR':>7}  Caption")
    print("-" * 110)
    for img in test_ids:
        refs = [simple_tokenize(c) for c in df_caps[df_caps["image"] == img]["caption"].tolist()]
        if args.flickr30k_hf:
            hyp = simple_tokenize(caption_pil_image(test_pil[img], encoder, decoder, vocab, device))
        else:
            import os as _os
            img_path = img if _os.path.isabs(img) else f"{args.images_dir}/{img}"
            hyp = simple_tokenize(caption_image(img_path, encoder, decoder, vocab, device))
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
                from PIL import Image as PILImage
                pil_img = PILImage.open(str(images_dir_abs / img)).convert("RGB")
                bleu_table.add_data(wandb.Image(pil_img), " ".join(hyp), ref_str, round(b1, 3), round(b4, 3), round(m, 3))

    cb1 = corpus_bleu(all_refs, all_hyps, weights=(1,0,0,0))
    cb4 = corpus_bleu(all_refs, all_hyps, weights=(.25,.25,.25,.25))
    cm  = float(np.mean(all_meteors))
    print("-" * 110)
    print(f"[bleu] Corpus BLEU-1: {cb1:.3f}  BLEU-4: {cb4:.3f}  METEOR: {cm:.3f}")

    if use_wandb:
        wandb.log({"bleu_eval_table": bleu_table, "bleu/corpus_bleu1": cb1, "bleu/corpus_bleu4": cb4, "bleu/meteor": cm})
    # ------------------------------------------------

    if use_wandb:
        wandb.finish() # tanca corectament la run de wandb perquè marqui l'experiment com acabat


if __name__ == "__main__":
    main()
