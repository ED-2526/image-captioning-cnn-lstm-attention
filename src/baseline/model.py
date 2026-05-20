"""CNN encoder + LSTM decoder for image captioning."""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


class EncoderCNN(nn.Module):
    """ResNet preentrenat → vector de mida encoder_size."""

    def __init__(self, encoder_size: int = 256, backbone: str = "resnet50", fine_tune_encoder: bool = False):
        super().__init__()
        if backbone == "resnet50":
            kw = dict(weights=models.ResNet50_Weights.IMAGENET1K_V2) if hasattr(models, "ResNet50_Weights") else dict(pretrained=True)
            net = models.resnet50(**kw)
        elif backbone == "resnet152":
            kw = dict(weights=models.ResNet152_Weights.IMAGENET1K_V2) if hasattr(models, "ResNet152_Weights") else dict(pretrained=True)
            net = models.resnet152(**kw)
        else:
            raise ValueError(f"Backbone no suportat: {backbone}")

        self.net = net
        self.fine_tune_encoder = fine_tune_encoder
        self.linear = nn.Linear(net.fc.in_features, encoder_size)  # 2048 → encoder_size
        self.bn = nn.BatchNorm1d(encoder_size, momentum=0.01)

        # Congelem tot el backbone
        for p in self.net.parameters():
            p.requires_grad = False

        # Si fine_tune_encoder, descongelem layer4 (últimes capes, les més útils)
        if fine_tune_encoder:
            for p in self.net.layer4.parameters():
                p.requires_grad = True

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Passem per les capes congelades sense guardar gradients
        with torch.set_grad_enabled(self.fine_tune_encoder and self.training):
            x = self.net.conv1(images)
            x = self.net.bn1(x)
            x = self.net.relu(x)
            x = self.net.maxpool(x)
            x = self.net.layer1(x)
            x = self.net.layer2(x)
            x = self.net.layer3(x)

        # layer4: amb gradients si fine_tune_encoder, sense si no
        if self.fine_tune_encoder and self.training:
            x = self.net.layer4(x)
            features = self.net.avgpool(x)
        else:
            with torch.no_grad():
                x = self.net.layer4(x)
                features = self.net.avgpool(x)

        features = features.flatten(1)              # [B, 2048]
        return self.bn(self.linear(features))       # [B, encoder_size]


class DecoderRNN(nn.Module):
    """LSTM que genera captions a partir del vector de la imatge."""

    def __init__(self, encoder_size: int, embed_size: int, hidden_size: int, vocab_size: int,
                 num_layers: int = 1, max_seq_length: int = 20, dropout: float = 0.5,
                 pretrained_weights: "torch.Tensor | None" = None, freeze_embeddings: bool = False):
        super().__init__()
        # encoder_size: mida del vector de la imatge (EncoderCNN output)
        # embed_size:   mida dels embeddings de paraules (GloVe → 300)
        # hidden_size:  mida de l'estat intern de la LSTM
        self.embed = nn.Embedding(vocab_size, embed_size)
        if pretrained_weights is not None:
            self.embed.weight = nn.Parameter(pretrained_weights)
        if freeze_embeddings:
            self.embed.weight.requires_grad = False
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seq_length = max_seq_length
        self.num_layers = num_layers
        # Projectem el vector de la imatge per inicialitzar h0 i c0
        # encoder_size pot ser diferent d'embed_size: les dues rutes són independents
        self.img_to_h = nn.Linear(encoder_size, hidden_size * num_layers)
        self.img_to_c = nn.Linear(encoder_size, hidden_size * num_layers)

    def _init_hidden(self, features: torch.Tensor):
        B = features.size(0)
        h0 = self.img_to_h(features).view(self.num_layers, B, -1)  # [num_layers, B, hidden]
        c0 = self.img_to_c(features).view(self.num_layers, B, -1)
        return h0, c0

    def forward(self, features: torch.Tensor, captions: torch.Tensor, lengths: list[int]):
        # features: [B, encoder_size]  captions: [B, T]
        states = self._init_hidden(features)
        embeddings = self.dropout(self.embed(captions))              # [B, T, embed_size]
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed, states)
        return self.linear(self.dropout(hiddens.data))               # [sum(lengths), vocab_size]

    def forward_scheduled(self, features: torch.Tensor, captions: torch.Tensor, p_teacher: float):
        """Forward pas a pas amb scheduled sampling.
        p_teacher=1.0 → teacher forcing pur
        p_teacher=0.0 → sempre predicció del model
        Cada exemple del batch decideix independentment a cada timestep.
        """
        B, T = captions.size()
        states = self._init_hidden(features)
        inp = captions[:, 0]  # <start>
        outputs = []
        for t in range(1, T):
            emb = self.dropout(self.embed(inp)).unsqueeze(1)         # [B, 1, embed_size]
            hiddens, states = self.lstm(emb, states)
            logits = self.linear(self.dropout(hiddens.squeeze(1)))   # [B, vocab_size]
            outputs.append(logits)
            # Màscara per-sample: cada exemple del batch decideix independentment
            use_teacher = torch.rand(B, device=features.device) < p_teacher  # [B]
            model_pred  = logits.argmax(dim=1).detach()              # [B]
            inp = torch.where(use_teacher, captions[:, t], model_pred)
        return torch.stack(outputs, dim=1)                           # [B, T-1, vocab_size]

    @torch.no_grad()
    def sample(self, features: torch.Tensor, states=None) -> torch.Tensor:
        """Generació greedy: a cada pas escull la paraula més probable."""
        sampled = []
        if states is None:
            states = self._init_hidden(features)
        inputs = self.embed(
            torch.ones(features.size(0), dtype=torch.long, device=features.device)
        ).unsqueeze(1)                                             # [B, 1, embed_size]
        for _ in range(self.max_seq_length):
            hiddens, states = self.lstm(inputs, states)
            predicted = self.linear(hiddens.squeeze(1)).max(1)[1]  # [B]
            sampled.append(predicted)
            inputs = self.embed(predicted).unsqueeze(1)
        return torch.stack(sampled, dim=1)                         # [B, max_seq_length]

    @torch.no_grad()
    def beam_search(self, features: torch.Tensor, beam_size: int = 3) -> list[int]:
        """Beam search per a una sola imatge (B=1). Retorna la millor seqüència."""
        device = features.device
        END_TOKEN = 2   # índex de <end>

        # Estats inicials i primer token <start>
        h, c = self._init_hidden(features)          # [num_layers, 1, hidden]
        start_embed = self.embed(
            torch.ones(1, dtype=torch.long, device=device)
        ).unsqueeze(1)                               # [1, 1, embed_size]
        hiddens, (h, c) = self.lstm(start_embed, (h, c))
        log_probs = torch.log_softmax(self.linear(hiddens.squeeze(1)), dim=1)  # [1, vocab]

        # Inicialitza beam: (score, seqüència, h, c)
        topk_scores, topk_ids = log_probs[0].topk(beam_size)
        beams = [
            (topk_scores[i].item(), [topk_ids[i].item()],
             h.clone(), c.clone())
            for i in range(beam_size)
        ]

        completed = []

        for _ in range(self.max_seq_length - 1):
            new_beams = []
            for score, seq, h_b, c_b in beams:
                if seq[-1] == END_TOKEN:
                    completed.append((score, seq))
                    continue
                last_word = torch.tensor([seq[-1]], device=device)
                inp = self.embed(last_word).unsqueeze(1)          # [1, 1, embed_size]
                hid, (h_new, c_new) = self.lstm(inp, (h_b, c_b))
                lp = torch.log_softmax(self.linear(hid.squeeze(1)), dim=1)  # [1, vocab]
                tk_scores, tk_ids = lp[0].topk(beam_size)
                for i in range(beam_size):
                    new_beams.append((
                        score + tk_scores[i].item(),
                        seq + [tk_ids[i].item()],
                        h_new.clone(), c_new.clone()
                    ))
            # Queda amb els beam_size millors
            new_beams.sort(key=lambda x: -x[0])
            beams = new_beams[:beam_size]

        # Afegeix els beams no acabats
        completed += [(s, sq) for s, sq, _, _ in beams]
        # Normalitza per longitud i retorna la millor seqüència
        completed.sort(key=lambda x: -x[0] / max(len(x[1]), 1))
        return completed[0][1]
