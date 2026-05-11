"""CNN encoder + LSTM decoder for image captioning."""
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
        return net, net.fc.in_features, "resnet"
    if backbone == "resnet152":
        try:
            weights = models.ResNet152_Weights.IMAGENET1K_V2 if pretrained else None
            net = models.resnet152(weights=weights)
        except (AttributeError, TypeError):
            net = models.resnet152(pretrained=pretrained)
        return net, net.fc.in_features, "resnet"
    if backbone == "efficientnet_b0":
        try:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.efficientnet_b0(weights=weights)
        except (AttributeError, TypeError):
            net = models.efficientnet_b0(pretrained=pretrained)
        return net, net.classifier[1].in_features, "efficientnet"
    if backbone == "vgg16":
        try:
            weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.vgg16(weights=weights)
        except (AttributeError, TypeError):
            net = models.vgg16(pretrained=pretrained)
        return net, net.classifier[0].in_features, "vgg"
    raise ValueError(f"Unsupported backbone: {backbone}")


class EncoderCNN(nn.Module):
    """Frozen CNN encoder that projects image features into `embed_size`."""

    def __init__(self, embed_size: int = 256, backbone: str = "resnet50", pretrained: bool = True):
        super().__init__()
        self.backbone = backbone
        self.net, feature_dim, self.backbone_kind = _load_backbone(backbone, pretrained=pretrained)
        self.linear = nn.Linear(feature_dim, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

        for param in self.net.parameters():
            param.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self.backbone_kind == "resnet":
                x = self.net.conv1(images)
                x = self.net.bn1(x)
                x = self.net.relu(x)
                x = self.net.maxpool(x)
                x = self.net.layer1(x)
                x = self.net.layer2(x)
                x = self.net.layer3(x)
                x = self.net.layer4(x)
                features = self.net.avgpool(x)
            elif self.backbone_kind == "efficientnet":
                x = self.net.features(images)
                features = self.net.avgpool(x)
            else:
                x = self.net.features(images)
                features = self.net.avgpool(x)
        features = features.flatten(1)
        return self.bn(self.linear(features))


class DecoderRNN(nn.Module):
    """LSTM decoder conditioned on image features."""

    def __init__(
        self,
        embed_size: int,
        hidden_size: int,
        vocab_size: int,
        num_layers: int = 1,
        max_seq_length: int = 20,
        dropout: float = 0.5,
        pretrained_weights: "torch.Tensor | None" = None,
        freeze_embeddings: bool = False,
        decoder_direction: str = "uni",
    ):
        super().__init__()
        if decoder_direction not in {"uni", "bidir"}:
            raise ValueError(f"Unsupported decoder_direction: {decoder_direction}")

        self.embed = nn.Embedding(vocab_size, embed_size)
        if pretrained_weights is not None:
            self.embed.weight = nn.Parameter(pretrained_weights)
        if freeze_embeddings:
            self.embed.weight.requires_grad = False

        self.decoder_direction = decoder_direction
        self.bidirectional = decoder_direction == "bidir"
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length
        self.dropout = nn.Dropout(dropout)

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=self.bidirectional,
        )
        out_dim = hidden_size * (2 if self.bidirectional else 1)
        self.linear = nn.Linear(out_dim, vocab_size)

    def forward(self, features: torch.Tensor, captions: torch.Tensor, lengths: list[int]):
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        return self.linear(self.dropout(hiddens.data))

    @torch.no_grad()
    def sample(self, features: torch.Tensor, states=None) -> torch.Tensor:
        if self.bidirectional:
            raise RuntimeError(
                "Bidirectional baseline decoders are training-only and do not support autoregressive sampling. "
                "Use --skip-test-captioning for these runs."
            )

        sampled = []
        inputs = features.unsqueeze(1)
        for _ in range(self.max_seq_length):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            sampled.append(predicted)
            inputs = self.embed(predicted).unsqueeze(1)
        return torch.stack(sampled, dim=1)
