"""Encoder CNN + Decoder LSTM per image captioning."""

import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    """ResNet50 preentrenat que extreu un vector de característiques de la imatge.

    La ResNet estava entrenada per classificar 1000 categories d'ImageNet.
    Eliminem la seva última capa (classificador) i afegim la nostra pròpia
    capa lineal per obtenir un vector de mida embed_size.

    Tots els pesos de la ResNet estan congelats (no s'entrenen).
    Només s'entrena la capa lineal final.
    """

    def __init__(self, encoder_size=256):
        super().__init__()

        # Carreguem ResNet50 preentrenada
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Eliminem la última capa (fc) i guardem la resta
        # list(resnet.children())[:-1] → tot excepte la capa fc final
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])

        # Capa lineal: 2048 (sortida de ResNet) → encoder_size
        self.linear = nn.Linear(resnet.fc.in_features, encoder_size)
        self.bn = nn.BatchNorm1d(encoder_size)

        # Congelar ResNet: no volem que els seus pesos canviïn
        for param in self.resnet.parameters():
            param.requires_grad = False

    def forward(self, images):
        # images: [B, 3, 224, 224]
        with torch.no_grad():
            features = self.resnet(images)        # [B, 2048, 1, 1]
        features = features.flatten(1)            # [B, 2048]
        features = self.bn(self.linear(features)) # [B, encoder_size]
        return features


class DecoderLSTM(nn.Module):
    """LSTM que genera captions paraula per paraula.

    La imatge s'usa per inicialitzar el hidden state de la LSTM (h0, c0).
    Així la LSTM ja "sap" de quina imatge parla des del primer pas,
    sense necessitar-la com a primer token de la seqüència.

    Funciona en dos modes:
    - Training (forward): rep la caption real com a input (teacher forcing)
    - Inferència (sample): genera la caption sola, paraula per paraula
    """

    def __init__(self, encoder_size, word_embed_size, hidden_size, vocab_size,
                 num_layers=1, pretrained_embeddings=None):
        super().__init__()

        # encoder_size:    mida del vector de la imatge (sortida de EncoderCNN)
        # word_embed_size: mida dels vectors de les paraules (pot ser diferent!)
        # hidden_size:     mida de l'estat intern de la LSTM

        self.embed = nn.Embedding(vocab_size, word_embed_size)
        # Si tenim GloVe, substituïm els pesos aleatoris pels preentrenats
        if pretrained_embeddings is not None:
            self.embed.weight = nn.Parameter(pretrained_embeddings)
        self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

        # Projectem el vector de la imatge [encoder_size] → [hidden_size]
        # per inicialitzar h0 i c0 de la LSTM
        self.img_to_h = nn.Linear(encoder_size, hidden_size)
        self.img_to_c = nn.Linear(encoder_size, hidden_size)

    def _init_hidden(self, features):
        """Converteix el vector de la imatge en (h0, c0) per inicialitzar la LSTM.

        features: [B, embed_size]
        retorna:  ([1, B, hidden_size], [1, B, hidden_size])
        """
        h0 = self.img_to_h(features).unsqueeze(0)  # [1, B, hidden_size]
        c0 = self.img_to_c(features).unsqueeze(0)  # [1, B, hidden_size]
        return h0, c0

    def forward(self, features, captions):
        """Mode training amb teacher forcing.

        Args:
            features: vector de la imatge [B, embed_size]
            captions: caption real sense l'últim token [B, T-1]
        Returns:
            prediccions [B, T-1, vocab_size]
        """
        # Usem la imatge per inicialitzar el hidden state
        states = self._init_hidden(features)

        # Convertim tokens a embeddings i passem per la LSTM
        embeddings = self.embed(captions)              # [B, T-1, embed_size]
        hiddens, _ = self.lstm(embeddings, states)     # [B, T-1, hidden_size]
        return self.linear(hiddens)                    # [B, T-1, vocab_size]

    def step(self, token, states):
        """Un sol pas de la LSTM: rep un token i retorna logits i nous estats."""
        inp = self.embed(token).unsqueeze(1)           # [B, 1, embed]
        hiddens, states = self.lstm(inp, states)       # [B, 1, hidden]
        logits = self.linear(hiddens.squeeze(1))       # [B, vocab]
        return logits, states

    @torch.no_grad()
    def sample(self, features, max_length=20):
        """Mode inferència: genera la caption greedy (paraula més probable a cada pas).

        Args:
            features: vector de la imatge [B, embed_size]
        Returns:
            [B, max_length] índexs de paraules generades
        """
        # Inicialitzem la LSTM amb el vector de la imatge
        states = self._init_hidden(features)

        # Primer input: token <start> (índex 1)
        inputs = torch.ones(features.size(0), dtype=torch.long, device=features.device)
        inputs = self.embed(inputs).unsqueeze(1)  # [B, 1, embed_size]

        generated = []
        for _ in range(max_length):
            hiddens, states = self.lstm(inputs, states)     # [B, 1, hidden_size]
            output = self.linear(hiddens.squeeze(1))        # [B, vocab_size]
            predicted = output.argmax(dim=1)                # [B]
            generated.append(predicted)
            inputs = self.embed(predicted).unsqueeze(1)     # [B, 1, embed_size]

        return torch.stack(generated, dim=1)  # [B, max_length]
