# XN-Image Captioning (Group 06)

This project implements two deep learning models capable of generating automatic captions for images, using the Flickr8k and Flickr30k datasets.

## Models

### Baseline — CNN + LSTM
- **Encoder:** ResNet-152 pretrained on ImageNet. Extracts a single global feature vector `[B, 256]` per image.
- **Decoder:** LSTM conditioned on the image feature, injected once as the first input token.
- **Embeddings:** GloVe 300d (`glove.6B.300d.txt`) for word representation.
- **Loss:** CrossEntropyLoss with label smoothing 0.1.
- **Generation:** Greedy decoding (picks the most probable word at each step).

### Attention — CNN + LSTM + Bahdanau Attention
- **Encoder:** ResNet-152 that outputs a spatial feature grid `[B, 49, 2048]` (7×7 regions). Fine-tuning of `layer4` from epoch 5.
- **Decoder:** LSTM with Bahdanau (additive) attention. At each step, computes which of the 49 image regions to focus on via learned attention weights (alpha).
- **Embeddings:** GloVe 300d (`glove.6B.300d.txt`) for word representation.
- **Loss:** CrossEntropyLoss with label smoothing 0.1 + Doubly Stochastic attention regularisation.
- **Generation:** Beam search (`beam_size=3`) for better caption quality.
- **Fine-tuning (optional):** SCST (Self-Critical Sequence Training) with CIDEr-D reward after CE training.

## Results

### Flickr8k test set (1000 images)
| Model | Corpus BLEU-1 | Corpus BLEU-4 |
|-------|--------------|--------------|
| Baseline | 0.568 | 0.176 |
| Attention | 0.634 | 0.215 |

### Flickr30k test set (1000 images)
| Model | Corpus BLEU-1 | Corpus BLEU-4 | METEOR |
|-------|--------------|--------------|--------|
| Attention (CE + Label Smoothing) | 0.669 | **0.253** | **0.412** |

## Code structure

```
├── src/
│   ├── shared/
│   │   ├── dataset.py        # Flickr8kDataset, Flickr30kHFDataset, DataLoader, transforms, splits
│   │   ├── vocabulary.py     # Vocabulary class, tokenizer, build_vocab, GloVe/Word2Vec loaders
│   │   └── losses.py         # SemanticCrossEntropyLoss, build_soft_labels
│   ├── baseline/
│   │   ├── model.py          # EncoderCNN + DecoderRNN
│   │   ├── train.py          # Training loop (baseline)
│   │   └── sample.py         # Caption generation (greedy)
│   └── attention/
│       ├── model.py          # EncoderCNNAttention + Attention + AttentionDecoder
│       ├── train.py          # Training loop (attention) + SCST fine-tuning
│       └── sample.py         # Caption generation (beam search)
│
├── checkpoints/                    # Saved baseline model checkpoints (Flickr8k)
├── checkpoints_attention/          # Saved attention model checkpoints (Flickr8k)
├── checkpoints_attention_30k/      # Saved attention model checkpoints (Flickr30k)
├── checkpoints_attention_scst_30k/ # Saved SCST fine-tuned checkpoints (Flickr30k)
│
├── dataset/
│   ├── flickr30k_hf/         # Flickr30k HuggingFace cache (nlphuji/flickr30k)
│   ├── vocab_flickr30k.pkl   # Prebuilt vocabulary Flickr30k (threshold=5, ~2982 words)
│   ├── vocab.pkl             # Prebuilt vocabulary Flickr8k (threshold=5, ~2982 words)
│   └── glove.6B.300d.txt     # GloVe embeddings 300d
│
└── requirements.txt
```

## Dataset

**Flickr8k** — 8091 images with 5 human-written captions each (40455 total).

| Split | Images | Captions |
|-------|--------|----------|
| Train | 6091 (75.3%) | 30455 |
| Val | 1000 (12.4%) | 5000 |
| Test | 1000 (12.4%) | 5000 |

**Flickr30k** — 31783 images with 5 human-written captions each. Loaded via HuggingFace (`nlphuji/flickr30k`).

| Split | Images | Captions |
|-------|--------|----------|
| Train | 29783 | 148915 |
| Val | 1000 | 5000 |
| Test | 1000 | 5000 |

## Usage

**Train baseline (Flickr8k):**
```bash
python -m src.baseline.train \
  --backbone resnet152 \
  --glove-path dataset/glove.6B.300d.txt \
  --epochs 10 --batch-size 32 --wandb
```

**Train attention (Flickr30k, best configuration):**
```bash
python -m src.attention.train \
  --flickr30k-hf \
  --flickr30k-hf-cache dataset/flickr30k_hf \
  --vocab-path dataset/vocab_flickr30k.pkl \
  --backbone resnet152 \
  --glove-path dataset/glove.6B.300d.txt \
  --no-semantic-loss \
  --epochs 30 --batch-size 32 \
  --hidden-size 512 --attention-dim 256 --dropout 0.5 \
  --lr 1e-3 --finetune-cnn-epoch 5 \
  --patience 7 --wandb
```

**SCST fine-tuning from best CE checkpoint:**
```bash
python -m src.attention.train \
  --flickr30k-hf \
  --flickr30k-hf-cache dataset/flickr30k_hf \
  --vocab-path dataset/vocab_flickr30k.pkl \
  --backbone resnet152 \
  --glove-path dataset/glove.6B.300d.txt \
  --no-semantic-loss \
  --epochs 0 \
  --scst-epochs 10 --scst-lr 1e-5 --scst-batch-size 32 \
  --scst-checkpoint checkpoints_attention_30k/ckpt_best.pt \
  --scst-checkpoints-dir checkpoints_attention_scst_30k \
  --wandb
```

**Generate caption for a single image:**
```bash
python -m src.attention.sample \
  --image path/to/image.jpg \
  --checkpoint checkpoints_attention_30k/ckpt_best.pt \
  --vocab dataset/vocab_flickr30k.pkl \
  --beam-size 3
```

## Contributors

Alicia Martí (AliciaMartiL@autonoma.cat), Maria Siles (Maria.Siles@autonoma.cat), Oriol Vilà (Oriol.VilaSa@autonoma.cat) and Clara Priego (Clara.PriegoF@autonoma.cat)

Xarxes Neuronals i Aprenentatge Profund — Grau d'Enginyeria de Dades, UAB, 2026
