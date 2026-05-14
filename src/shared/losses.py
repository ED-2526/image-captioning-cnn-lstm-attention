"""Custom loss functions per image captioning."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_glove_similarity(glove_weights: torch.Tensor) -> torch.Tensor:
    """Construeix una matriu de similitud cosinus entre totes les paraules del vocabulari.

    La similitud cosinus entre dos vectors mesura com de similars són ignorant la seva magnitud.
    Resultat: matriu [vocab_size, vocab_size] on sim[i][j] = similitud entre paraula i i paraula j.
    Exemple: sim["dog"]["puppy"] ≈ 0.9, sim["dog"]["car"] ≈ 0.1

    Args:
        glove_weights: matriu de pesos GloVe [vocab_size, glove_dim]
    Returns:
        matriu de similitud [vocab_size, vocab_size], valors entre -1 i 1
    """
    # Normalitzem cada vector a longitud 1 perquè el producte escalar sigui igual a la similitud cosinus
    w = F.normalize(glove_weights, dim=1)  # [vocab_size, glove_dim], cada fila té norma 1
    # Producte matricial: w[i] · w[j] = similitud cosinus entre paraula i i paraula j
    return w @ w.t()  # [vocab_size, vocab_size]


class SemanticCrossEntropyLoss(nn.Module):
    """CrossEntropy amb soft labels basades en TOTA la matriu de similitud GloVe.

    Diferència respecte a CrossEntropy normal:
    - CrossEntropy normal: el target és una paraula concreta (label hard, 0 o 1)
      Exemple: target="dog" → [0, 0, 0, 1, 0, 0, ...] (1 sol a la posició de "dog")
    - SemanticCE: el pes es distribueix entre TOTES les paraules proporcional a la seva similitud
      Exemple: target="dog" → [0, 0, 0.8, 0.05, 0.03, ...] ("puppy" rep 0.05 perquè s'assembla a "dog")

    Avantatge: el model no és penalitzat si prediu "puppy" quan el target és "dog",
    perquè semànticament són quasi equivalents.

    Inconvenient: distribuir el pes entre TOT el vocabulari (milers de paraules) pot
    introduir soroll. TopKSemanticLoss ho soluciona limitant-ho a les K més similars.
    """

    def __init__(self, soft_labels: torch.Tensor):
        """
        Args:
            soft_labels: matriu de similitud GloVe [vocab_size, vocab_size],
                         creada amb build_glove_similarity()
        """
        super().__init__()
        # register_buffer guarda la matriu com a paràmetre del mòdul però sense gradient
        # (no s'entrena, només s'usa per consultar similituds)
        self.register_buffer("soft_labels", soft_labels)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  prediccions del model [N, vocab_size], valors sense softmax
            targets: paraules target reals  [N], índexs enters
        Returns:
            loss escalar (mean sobre el batch)
        """
        # Convertim logits a log-probabilitats (log perquè la CE és -sum(target * log_prob))
        log_probs = torch.log_softmax(logits, dim=1)  # [N, vocab_size]

        # Per cada exemple del batch, agafem la fila de la matriu de similitud corresponent al target
        # Exemple: si target[0] = índex de "dog", soft_tgts[0] = similituds de "dog" amb totes les paraules
        soft_tgts = self.soft_labels[targets]  # [N, vocab_size]

        # Fórmula de cross-entropy amb soft labels: -sum(soft_target * log_prob)
        # .sum(dim=1): suma sobre el vocabulari → [N]
        # .mean(): mitjana sobre el batch → escalar
        return -(soft_tgts * log_probs).sum(dim=1).mean()


class TopKSemanticLoss(nn.Module):
    """CrossEntropy amb soft labels basades en les K paraules MÉS SIMILARS per GloVe.

    Millora de SemanticCrossEntropyLoss: en comptes de distribuir el pes entre
    tot el vocabulari, només ho fa entre les K paraules més similars al target.

    Distribució del pes per cada target:
    - (1 - alpha) → a la paraula target real         (exemple: 0.8 a "dog")
    - alpha       → repartit entre les K més similars (exemple: 0.2 entre "puppy","hound",...)
    - 0           → a la resta de paraules del vocabulari

    Exemple amb k=10, alpha=0.2, target="dog":
      "dog"   → 0.80 (la paraula real)
      "puppy" → 0.05 (la més similar)
      "hound" → 0.04
      ...     → (les altres 8 del top-10)
      "car"   → 0.00 (fora del top-10)

    Avantatge respecte a SemanticCE: evita el soroll de distribuir pes a paraules
    poc relacionades (ex: "dog" i "mathematics" tenen similitud ~0.01 però no zero).
    """

    def __init__(self, soft_labels: torch.Tensor, k: int = 10, alpha: float = 0.2):
        """
        Args:
            soft_labels: matriu de similitud GloVe [vocab_size, vocab_size]
            k:     nombre de paraules similars a considerar (recomanat: 5-20)
            alpha: pes total donat a les K paraules similars (1-alpha va al target real)
                   alpha=0.0 → equivalent a CrossEntropy normal (sense suavitzat)
                   alpha=0.2 → 20% del pes va a similars, 80% al target real
                   alpha=1.0 → tot el pes va a similars (no recomanat)
        """
        super().__init__()
        self.k = k
        self.alpha = alpha

        # Precomputem les K+1 paraules més similars per cada paraula del vocabulari
        # +1 perquè la paraula més similar a si mateixa és ella mateixa (la descartarem amb scatter_)
        topk_vals, topk_idx = soft_labels.topk(k + 1, dim=1)  # ambdós [vocab_size, k+1]

        # Normalitzem els valors de similitud perquè sumin 1 (distribució de probabilitat)
        # Exemple: similituds [0.9, 0.8, 0.7] → softmax → [0.40, 0.33, 0.27]
        topk_vals = torch.softmax(topk_vals, dim=1)

        # Guardem com a buffers (no s'entrenen, no canvien durant l'entrenament)
        self.register_buffer("topk_idx", topk_idx)    # índexs de les K+1 paraules més similars [vocab_size, k+1]
        self.register_buffer("topk_vals", topk_vals)  # pesos normalitzats de les K+1 similars [vocab_size, k+1]

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  prediccions del model [N, vocab_size], valors sense softmax
            targets: paraules target reals  [N], índexs enters
        Returns:
            loss escalar (mean sobre el batch)
        """
        log_probs = torch.log_softmax(logits, dim=1)  # [N, vocab_size]

        # Creem la matriu de soft labels buida (tots zeros)
        soft_tgts = torch.zeros_like(logits)  # [N, vocab_size]

        # Pas 1: posem el pes alpha repartit entre les K paraules més similars
        # topk_idx[targets]: per cada exemple del batch, els índexs de les K+1 paraules més similars
        # topk_vals[targets] * alpha: els pesos normalitzats, escalats per alpha
        # scatter_ escriu aquests pesos a les posicions corresponents de soft_tgts
        soft_tgts.scatter_(1, self.topk_idx[targets], self.topk_vals[targets] * self.alpha)

        # Pas 2: afegim el pes (1-alpha) a la paraula target real
        # scatter_add_ suma (en comptes de sobreescriure) perquè el target ja pot estar al top-K
        # targets.unsqueeze(1): [N] → [N, 1] per poder usar-lo amb scatter_add_
        soft_tgts.scatter_add_(1, targets.unsqueeze(1),
                               torch.ones(len(targets), 1, device=logits.device) * (1 - self.alpha))

        # Fórmula CE: -sum(soft_target * log_prob), sumem sobre vocab i fem mean sobre batch
        return -(soft_tgts * log_probs).sum(dim=1).mean()
