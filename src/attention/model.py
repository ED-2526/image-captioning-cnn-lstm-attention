"""CNN encoder with spatial features + attention decoder."""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


def _load_backbone(backbone: str, pretrained: bool = True):
    if backbone == "resnet50":
        try:
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            net = models.resnet50(weights=weights)
        except (AttributeError, TypeError):
            net = models.resnet50(pretrained=pretrained)
        return net, 2048, "resnet"
    if backbone == "resnet152":
        try:
            weights = models.ResNet152_Weights.IMAGENET1K_V2 if pretrained else None
            net = models.resnet152(weights=weights)
        except (AttributeError, TypeError):
            net = models.resnet152(pretrained=pretrained)
        return net, 2048, "resnet"
    if backbone == "efficientnet_b0":
        try:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.efficientnet_b0(weights=weights)
        except (AttributeError, TypeError):
            net = models.efficientnet_b0(pretrained=pretrained)
        return net, 1280, "efficientnet"
    if backbone == "vgg16":
        try:
            weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.vgg16(weights=weights)
        except (AttributeError, TypeError):
            net = models.vgg16(pretrained=pretrained)
        return net, 512, "vgg"
    raise ValueError(f"Unsupported backbone: {backbone}")


class EncoderCNNAttention(nn.Module):
    """CNN encoder that outputs a spatial grid of features for attention."""

    def __init__(self, backbone: str = "resnet50", pretrained: bool = True):
        super().__init__()
        self.backbone = backbone
        self.net, self.encoder_dim, self.backbone_kind = _load_backbone(backbone, pretrained=pretrained)

        for param in self.net.parameters():
            param.requires_grad = False
        self.finetuning = False

        if self.backbone_kind == "efficientnet":
            self._eff_prefix = self.net.features[:-2]
            self._eff_tunable = self.net.features[-2:]
        elif self.backbone_kind == "vgg":
            self._vgg_prefix = self.net.features[:-7]
            self._vgg_tunable = self.net.features[-7:]

    def enable_finetuning(self):
        self.finetuning = True
        if self.backbone_kind == "resnet":
            modules = [self.net.layer4]
        elif self.backbone_kind == "efficientnet":
            modules = [self._eff_tunable]
        else:
            modules = [self._vgg_tunable]
        params = []
        for module in modules:
            for param in module.parameters():
                param.requires_grad = True
                params.append(param)
        return params

    def _resnet_base_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        return x

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.backbone_kind == "resnet":
            if self.finetuning:
                with torch.no_grad():
                    x = self._resnet_base_forward(images)
                out = self.net.layer4(x)
            else:
                with torch.no_grad():
                    x = self._resnet_base_forward(images)
                    out = self.net.layer4(x)
        elif self.backbone_kind == "efficientnet":
            if self.finetuning:
                with torch.no_grad():
                    x = self._eff_prefix(images)
                out = self._eff_tunable(x)
            else:
                with torch.no_grad():
                    out = self.net.features(images)
        else:
            if self.finetuning:
                with torch.no_grad():
                    x = self._vgg_prefix(images)
                out = self._vgg_tunable(x)
            else:
                with torch.no_grad():
                    out = self.net.features(images)

        bsz, channels, height, width = out.shape
        out = out.permute(0, 2, 3, 1)
        return out.view(bsz, height * width, channels)


class Attention(nn.Module):
    """Bahdanau attention over encoder spatial features."""

    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int):
        super().__init__()
        self.enc_att = nn.Linear(encoder_dim, attention_dim)
        self.dec_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out: torch.Tensor, h: torch.Tensor):
        scores = self.full_att(
            torch.tanh(self.enc_att(encoder_out) + self.dec_att(h).unsqueeze(1))
        ).squeeze(2)
        alpha = self.softmax(scores)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return context, alpha


class AttentionDecoder(nn.Module):
    """Attention decoder with optional bidirectional training-only contextualization."""

    def __init__(
        self,
        encoder_dim: int = 2048,
        embed_size: int = 256,
        hidden_size: int = 512,
        vocab_size: int = 10000,
        attention_dim: int = 256,
        dropout: float = 0.5,
        max_seq_length: int = 20,
        pretrained_weights: "torch.Tensor | None" = None,
        freeze_embeddings: bool = False,
        decoder_direction: str = "uni",
    ):
        super().__init__()
        if decoder_direction not in {"uni", "bidir"}:
            raise ValueError(f"Unsupported decoder_direction: {decoder_direction}")

        self.encoder_dim = encoder_dim
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.decoder_direction = decoder_direction
        self.bidirectional = decoder_direction == "bidir"

        self.attention = Attention(encoder_dim, hidden_size, attention_dim)
        self.embed = nn.Embedding(vocab_size, embed_size)
        if pretrained_weights is not None:
            self.embed.weight = nn.Parameter(pretrained_weights)
        if freeze_embeddings:
            self.embed.weight.requires_grad = False

        self.dropout = nn.Dropout(dropout)
        self.lstm_cell = nn.LSTMCell(embed_size + encoder_dim, hidden_size)
        self.init_h = nn.Linear(encoder_dim, hidden_size)
        self.init_c = nn.Linear(encoder_dim, hidden_size)
        if self.bidirectional:
            self.context_rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(hidden_size * 2, vocab_size)
        else:
            self.context_rnn = None
            self.fc = nn.Linear(hidden_size, vocab_size)

    def _init_hidden(self, encoder_out: torch.Tensor):
        mean = encoder_out.mean(dim=1)
        return torch.tanh(self.init_h(mean)), torch.tanh(self.init_c(mean))

    def forward(self, encoder_out: torch.Tensor, captions: torch.Tensor, lengths: list[int]):
        batch_size = encoder_out.size(0)
        num_pixels = encoder_out.size(1)
        embeddings = self.dropout(self.embed(captions))
        h, c = self._init_hidden(encoder_out)

        decode_lengths = [length - 1 for length in lengths]
        max_t = max(decode_lengths)
        alphas_sum = torch.zeros(batch_size, num_pixels, device=encoder_out.device)

        hidden_seq = None
        preds = []
        if self.bidirectional:
            hidden_seq = torch.zeros(batch_size, max_t, self.hidden_size, device=encoder_out.device)

        for t in range(max_t):
            bt = sum(1 for length in decode_lengths if length > t)
            context, alpha = self.attention(encoder_out[:bt], h[:bt])
            alphas_sum[:bt] = alphas_sum[:bt] + alpha
            lstm_in = torch.cat([embeddings[:bt, t], context], dim=1)
            h_new, c_new = self.lstm_cell(lstm_in, (h[:bt], c[:bt]))

            if self.bidirectional:
                hidden_seq[:bt, t] = h_new
            else:
                preds.append(self.fc(self.dropout(h_new)))

            if bt < batch_size:
                h = torch.cat([h_new, h[bt:]], dim=0)
                c = torch.cat([c_new, c[bt:]], dim=0)
            else:
                h, c = h_new, c_new

        if self.bidirectional:
            packed_hidden = pack_padded_sequence(hidden_seq, decode_lengths, batch_first=True)
            contextual_hidden, _ = self.context_rnn(packed_hidden)
            outputs = self.fc(self.dropout(contextual_hidden.data))
        else:
            outputs = torch.cat(preds, dim=0)

        return outputs, alphas_sum

    @torch.no_grad()
    def beam_search(
        self,
        encoder_out: torch.Tensor,
        start_idx: int,
        end_idx: int,
        beam_size: int = 3,
    ) -> list[int]:
        if self.bidirectional:
            raise RuntimeError(
                "Bidirectional attention decoders are training-only and do not support autoregressive caption generation. "
                "Use --skip-test-captioning for these runs."
            )

        device = encoder_out.device
        k = beam_size

        enc = encoder_out.expand(k, -1, -1)
        h, c = self._init_hidden(enc)
        seqs = torch.full((k, 1), start_idx, dtype=torch.long, device=device)
        scores = torch.zeros(k, device=device)
        complete_seqs, complete_scores = [], []

        for step in range(self.max_seq_length):
            embeddings = self.embed(seqs[:, -1])
            context, _ = self.attention(enc, h)
            lstm_in = torch.cat([embeddings, context], dim=1)
            h, c = self.lstm_cell(lstm_in, (h, c))

            log_probs = torch.log_softmax(self.fc(h), dim=1)
            total = scores.unsqueeze(1) + log_probs

            if step == 0:
                top_scores, top_words = total[0].topk(k)
            else:
                top_scores, top_words = total.view(-1).topk(k)

            beam_idx = top_words // self.vocab_size
            word_idx = top_words % self.vocab_size

            seqs = torch.cat([seqs[beam_idx], word_idx.unsqueeze(1)], dim=1)
            h, c = h[beam_idx], c[beam_idx]
            enc = enc[beam_idx]
            scores = top_scores

            still_running = []
            for j in range(k):
                if word_idx[j].item() == end_idx:
                    complete_seqs.append(seqs[j, 1:-1].tolist())
                    complete_scores.append(scores[j].item())
                else:
                    still_running.append(j)

            if not still_running:
                break

            k = len(still_running)
            seqs = seqs[still_running]
            h, c = h[still_running], c[still_running]
            enc = enc[still_running]
            scores = scores[still_running]

        if not complete_seqs:
            complete_seqs = [seqs[0, 1:].tolist()]
            complete_scores = [scores[0].item()]

        best = max(range(len(complete_scores)), key=lambda idx: complete_scores[idx])
        return complete_seqs[best]
