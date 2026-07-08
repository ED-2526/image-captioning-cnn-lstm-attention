from __future__ import annotations

import argparse
import pickle
import re
from collections import Counter
from pathlib import Path

import pandas as pd

PAD, START, END, UNK = "<pad>", "<start>", "<end>", "<unk>"


def simple_tokenize(text: str) -> list[str]:
    """Lowercase and split captions into simple word tokens."""
    return re.findall(r"[a-z0-9']+", text.lower())


class Vocabulary:
    """Bidirectional token/index mapping used by the captioning models."""

    def __init__(self):
        self.word2idx: dict[str, int] = {}
        self.idx2word: dict[int, str] = {}
        self.idx = 0
        for token in (PAD, START, END, UNK):
            self.add_word(token)

    def add_word(self, word: str) -> None:
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word: str) -> int:
        return self.word2idx.get(word, self.word2idx[UNK])

    def __len__(self) -> int:
        return len(self.word2idx)

    def encode(self, caption: str, add_special: bool = True) -> list[int]:
        ids = [self(token) for token in simple_tokenize(caption)]
        if add_special:
            ids = [self(START)] + ids + [self(END)]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        words = []
        for token_id in ids:
            word = self.idx2word.get(int(token_id), UNK)
            if skip_special and word in (PAD, START):
                continue
            if skip_special and word == END:
                break
            words.append(word)
        return " ".join(words)


def load_glove_weights(glove_path: str | Path, vocab: Vocabulary) -> tuple["torch.Tensor", int]:
    """Load GloVe vectors and build an embedding matrix aligned to `vocab`."""
    import torch

    print(f"[glove] loading {glove_path}...")
    glove: dict[str, list[float]] = {}
    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            glove[parts[0]] = [float(x) for x in parts[1:]]

    glove_dim = len(next(iter(glove.values())))
    weights = torch.randn(len(vocab), glove_dim) * 0.01
    weights[0] = 0

    found = 0
    for word, idx in vocab.word2idx.items():
        if word in glove:
            weights[idx] = torch.tensor(glove[word])
            found += 1

    print(f"[glove] matched {found}/{len(vocab)} vocabulary words ({glove_dim}d)")
    return weights, glove_dim


def load_word2vec_weights(
    w2v_path: str | Path,
    vocab: Vocabulary,
    binary: bool | None = None,
) -> tuple["torch.Tensor", int]:
    """Load Word2Vec vectors and build an embedding matrix aligned to `vocab`."""
    import torch
    from gensim.models import KeyedVectors

    w2v_path = Path(w2v_path)
    if binary is None:
        binary = w2v_path.suffix == ".bin"

    print(f"[word2vec] loading {w2v_path} (binary={binary})...")
    wv = KeyedVectors.load_word2vec_format(str(w2v_path), binary=binary)

    weights = torch.randn(len(vocab), wv.vector_size) * 0.01
    weights[0] = 0

    found = 0
    for word, idx in vocab.word2idx.items():
        if word in wv:
            weights[idx] = torch.tensor(wv[word])
            found += 1

    print(f"[word2vec] matched {found}/{len(vocab)} vocabulary words ({wv.vector_size}d)")
    return weights, wv.vector_size


def load_fasttext_weights(ft_path: str | Path, vocab: Vocabulary) -> tuple["torch.Tensor", int]:
    """Load FastText text vectors and build an embedding matrix aligned to `vocab`."""
    import torch

    ft_path = Path(ft_path)
    print(f"[fasttext] loading {ft_path}...")

    vectors: dict[str, list[float]] = {}
    with open(ft_path, encoding="utf-8", errors="ignore") as f:
        first = f.readline().strip().split()
        has_header = len(first) == 2 and all(part.isdigit() for part in first)
        if not has_header and first:
            vectors[first[0]] = [float(x) for x in first[1:]]

        for line in f:
            parts = line.rstrip().split()
            if len(parts) > 2:
                vectors[parts[0]] = [float(x) for x in parts[1:]]

    vector_dim = len(next(iter(vectors.values())))
    weights = torch.randn(len(vocab), vector_dim) * 0.01
    weights[0] = 0

    found = 0
    for word, idx in vocab.word2idx.items():
        if word in vectors:
            weights[idx] = torch.tensor(vectors[word])
            found += 1

    print(f"[fasttext] matched {found}/{len(vocab)} vocabulary words ({vector_dim}d)")
    return weights, vector_dim


def build_vocab(captions_csv: str | Path, threshold: int = 5) -> Vocabulary:
    df = pd.read_csv(captions_csv)
    counter: Counter[str] = Counter()
    for caption in df["caption"].astype(str):
        counter.update(simple_tokenize(caption))

    vocab = Vocabulary()
    for word, count in counter.items():
        if count >= threshold:
            vocab.add_word(word)
    return vocab


def build_vocab_hf(hf_dataset, threshold: int = 5) -> Vocabulary:
    """Build a vocabulary from the HuggingFace `nlphuji/flickr30k` dataset."""
    counter: Counter[str] = Counter()
    for row in hf_dataset["test"]:
        for caption in row["caption"]:
            counter.update(simple_tokenize(caption))

    vocab = Vocabulary()
    for word, count in counter.items():
        if count >= threshold:
            vocab.add_word(word)
    return vocab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions", default="dataset/captions.txt")
    parser.add_argument("--out", default="dataset/vocab.pkl")
    parser.add_argument("--threshold", type=int, default=5)
    args = parser.parse_args()

    vocab = build_vocab(args.captions, args.threshold)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(vocab, f)
    print(f"Vocab size: {len(vocab)} (threshold={args.threshold})")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
