"""Vocabulari: converteix paraules a números i viceversa."""

import re
from collections import Counter

import pandas as pd

# Tokens especials
PAD = "<pad>"    # 0 - padding (omplir fins la mateixa longitud)
START = "<start>"  # 1 - inici de frase
END = "<end>"    # 2 - fi de frase
UNK = "<unk>"    # 3 - paraula desconeguda


def tokenize(text):
    """'A dog runs!' → ['a', 'dog', 'runs']"""
    return re.findall(r"[a-z0-9']+", text.lower())


class Vocabulary:
    def __init__(self):
        # Diccionari paraula → índex
        self.word2idx = {}
        # Diccionari índex → paraula
        self.idx2word = {}
        self._idx = 0
        # Afegim els 4 tokens especials al principi
        for tok in (PAD, START, END, UNK):
            self._add(tok)

    def _add(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self._idx
            self.idx2word[self._idx] = word 
            self._idx += 1

    def encode(self, caption):
        """'a dog runs' → [1, 4, 27, 83, 2]  (amb <start> i <end>)"""
        tokens = tokenize(caption)
        return [self(START)] + [self(t) for t in tokens] + [self(END)]

    def decode(self, ids):
        """[1, 4, 27, 83, 2] → 'a dog runs'"""
        words = []
        for i in ids:
            word = self.idx2word.get(int(i), UNK)
            if word == END:
                break
            if word in (PAD, START):
                continue
            words.append(word)
        return " ".join(words)


def build_vocab(captions_csv, threshold=5):
    """Construeix el vocabulari a partir del CSV de captions.

    Només inclou paraules que apareixen >= threshold vegades.
    Les paraules rares es convertiran a <unk> durant l'entrenament.
    """
    df = pd.read_csv(captions_csv)

    # Comptem quantes vegades apareix cada paraula
    counter = Counter()
    for caption in df["caption"].astype(str):
        counter.update(tokenize(caption))

    # Construïm el vocabulari amb les paraules freqüents
    vocab = Vocabulary()
    for word, count in counter.items():
        if count >= threshold:
            vocab._add(word)

    return vocab


def load_glove(glove_path, vocab, glove_dim=300):
    """Carrega vectors GloVe i retorna una matriu [vocab_size, glove_dim].

    Les paraules del vocab que no estan a GloVe s'inicialitzen aleatòriament.
    """
    import torch

    # Llegim el fitxer GloVe: cada línia és "paraula v1 v2 v3 ..."
    print(f"[glove] carregant {glove_path}...")
    glove = {}
    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            glove[parts[0]] = [float(x) for x in parts[1:]]

    # Matriu de pesos inicialitzada aleatòriament (petita)
    weights = torch.randn(len(vocab), glove_dim) * 0.01
    weights[0] = 0  # <pad> sempre és vector zero

    # Substituïm els vectors aleatoris pels vectors GloVe quan existeixen
    found = 0
    for word, idx in vocab.word2idx.items():
        if word in glove:
            weights[idx] = torch.tensor(glove[word])
            found += 1

    print(f"[glove] {found}/{len(vocab)} paraules trobades ({glove_dim}d)")
    return weights
