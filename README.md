# image-captioning-cnn-lstm-attention

Deep learning project for automatic image caption generation with CNN encoders, LSTM decoders, and an attention-based extension. The repository contains a baseline CNN+LSTM model and a CNN+LSTM model with Bahdanau attention, trained and evaluated on Flickr-style captioning datasets.

Collaborative academic project developed for the Deep Learning course at UAB. Contributors: Alicia Martí, Maria Siles, Clara Priego and Oriol Vilà.

## What It Solves

Image captioning combines computer vision and natural language generation: given an input image, the model predicts a short natural-language description. This project compares a simple encoder-decoder baseline against an attention model that can focus on different spatial regions of the image while generating each word.

## Architecture

### Baseline: CNN + LSTM

- Encoder: ResNet-152 pretrained on ImageNet.
- Visual representation: one global image feature vector.
- Decoder: LSTM conditioned on the image feature.
- Embeddings: optional pretrained GloVe, Word2Vec or FastText vectors.
- Training objective: cross-entropy, with optional label smoothing or semantic loss.
- Decoding: greedy decoding or beam search support.

### Attention: CNN + LSTM + Bahdanau Attention

- Encoder: ResNet-152 spatial feature grid.
- Attention: additive attention over image regions at every decoding step.
- Decoder: LSTM receiving the attended visual context and word embedding.
- Training objective: cross-entropy with optional label smoothing and doubly stochastic attention regularization.
- Decoding: beam search.
- Optional fine-tuning: CNN layer unfreezing after a chosen epoch and SCST-style fine-tuning support.

## Results

### Flickr8k Test Set

| Model | Corpus BLEU-1 | Corpus BLEU-4 |
| --- | ---: | ---: |
| Baseline | 0.568 | 0.176 |
| Attention | 0.634 | 0.215 |

### Flickr30k Test Set

| Model | Corpus BLEU-1 | Corpus BLEU-4 | METEOR |
| --- | ---: | ---: | ---: |
| Attention, CE + Label Smoothing | 0.669 | **0.253** | **0.412** |

Example output format:

```text
Image: path/to/image.jpg
Generated caption: a dog runs through the grass
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The models use PyTorch and torchvision. For GPU training, install the PyTorch build that matches your CUDA version if the default pip package is not appropriate for your machine.

## Data And Weights

Datasets, pretrained word vectors, and checkpoints are intentionally not tracked in Git.

Expected local layout for Flickr8k-style training:

```text
dataset/
├── Images/
│   └── *.jpg
├── captions.txt
├── vocab.pkl                # optional; built automatically if missing
└── glove.6B.300d.txt        # optional pretrained embeddings
```

For Flickr30k, the training scripts can load `nlphuji/flickr30k` through HuggingFace Datasets and cache it under `dataset/flickr30k_hf`.

Checkpoint folders such as `checkpoints/`, `checkpoints_attention/`, and `checkpoints_attention_30k/` are ignored by Git. If you publish trained weights, host them separately and link them from a release page or model registry.

## Usage

Train the baseline model on a local Flickr8k-style dataset:

```bash
python -m src.baseline.train \
  --images-dir dataset/Images \
  --captions-csv dataset/captions.txt
```

Train the attention model:

```bash
python -m src.attention.train \
  --images-dir dataset/Images \
  --captions-csv dataset/captions.txt
```

Train the attention model with Flickr30k from HuggingFace:

```bash
python -m src.attention.train \
  --flickr30k-hf \
  --flickr30k-hf-cache dataset/flickr30k_hf
```

Use GloVe embeddings:

```bash
python -m src.attention.train \
  --images-dir dataset/Images \
  --captions-csv dataset/captions.txt \
  --glove-path dataset/glove.6B.300d.txt
```

Generate a caption from a trained attention checkpoint:

```bash
python -m src.attention.sample \
  --image path/to/image.jpg \
  --checkpoint checkpoints_attention/ckpt_best.pt \
  --vocab dataset/vocab.pkl
```

Weights & Biases logging is optional and only enabled when `--wandb` is passed.

## Repository Structure

```text
.
├── src/
│   ├── shared/       # datasets, vocabulary utilities, losses
│   ├── baseline/     # CNN+LSTM model, training, sampling
│   └── attention/    # CNN+LSTM+attention model, training, sampling
├── archive/          # historical academic starter/simple implementations
├── README.md
├── requirements.txt
└── .gitignore
```

The maintained public implementation lives in `src/`. Files under `archive/` are kept for academic provenance and are not the recommended entrypoints.

## Limitations

- The project targets research and coursework reproducibility, not production serving.
- Caption quality depends heavily on dataset size, vocabulary coverage, and checkpoint quality.
- BLEU and METEOR are useful comparison metrics but do not fully capture human caption quality.
- Large datasets, pretrained embeddings, and checkpoints must be downloaded or hosted separately.
- Inference examples require a compatible vocabulary file and checkpoint generated by the training scripts.

## Credits

Collaborative academic project developed for the Deep Learning course at UAB.

Contributors: Alicia Martí, Maria Siles, Clara Priego and Oriol Vilà.

My contributions included model implementation, training experiments, evaluation and documentation.
