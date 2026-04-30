# XN-Image Captioning (Group 06)

This project implements two deep learning models capable of generating automatic captions for images, using the Flickr8k dataset.

## Models

### Baseline — CNN + LSTM
- **Encoder:** ResNet-50 pretrained on ImageNet. Extracts a single feature vector `[B, 256]` per image.
- **Decoder:** LSTM conditioned on the image feature, injected once as the first input token.
- **Generation:** Greedy decoding (picks the most probable word at each step).

### Attention — CNN + LSTM + Bahdanau Attention
- **Encoder:** ResNet-50 that outputs a spatial feature grid `[B, 49, 2048]` (7×7 regions).
- **Decoder:** LSTM with Bahdanau attention. At each step, computes which image regions to focus on.
- **Generation:** Beam search (`beam_size=3`) for better caption quality.

## Results (Flickr8k test set, 1000 images)

| Model | Corpus BLEU-1 | Corpus BLEU-4 |
|---|---|---|
| Baseline | — | — |
| **Attention** | **0.634** | **0.215** |

## Code structure

```
├── src/
│   ├── shared/
│   │   ├── dataset.py        # Flickr8kDataset, DataLoader, transforms, splits
│   │   └── vocabulary.py     # Vocabulary class, tokenizer, build_vocab
│   ├── baseline/
│   │   ├── model.py          # EncoderCNN + DecoderRNN
│   │   ├── train.py          # Training loop (baseline)
│   │   └── sample.py         # Caption generation (greedy)
│   └── attention/
│       ├── model.py          # EncoderCNNAttention + AttentionDecoder + Attention
│       ├── train.py          # Training loop (attention)
│       └── sample.py         # Caption generation (beam search)
│
├── eval_bleu_baseline.py     # BLEU evaluation on test set (baseline model)
├── eval_bleu_attention.py    # BLEU evaluation on test set (attention model)
│
├── checkpoints/              # Saved baseline model checkpoints
├── checkpoints_attention/    # Saved attention model checkpoints
├── data/flickr8k/
│   ├── Images/               # 8091 .jpg images
│   ├── captions.txt          # CSV: image, caption (5 captions per image)
│   └── vocab.pkl             # Prebuilt vocabulary (threshold=5, 2982 words)
└── requirements.txt
```

## Dataset

**Flickr8k** — 8091 images with 5 human-written captions each (40455 total).

| Split | Images | Captions |
|---|---|---|
| Train | 6091 (75.3%) | 30455 |
| Val | 1000 (12.4%) | 5000 |
| Test | 1000 (12.4%) | 5000 |

Download from [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k) and place under `data/flickr8k/`.

## Usage

**Train baseline:**
```bash
python -m src.baseline.train --epochs 5 --batch-size 32 --wandb
```

**Train attention:**
```bash
python -m src.attention.train --epochs 10 --batch-size 32 --wandb
```

**Evaluate BLEU (baseline):**
```bash
python eval_bleu_baseline.py
```

**Evaluate BLEU (attention):**
```bash
python eval_bleu_attention.py
```

## Contributors

Alicia Martí (AliciaMartiL@autonoma.cat), Maria Siles (Maria.Siles@autonoma.cat), Oriol Vilà (Oriol.VilaSa@autonoma.cat) and Clara Priego (Clara.PriegoF@autonoma.cat)

Xarxes Neuronals i Aprenentatge Profund
Grau d'Enginyeria de Dades, UAB, 2026
