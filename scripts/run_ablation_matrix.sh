#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Backward-compatible entrypoint. The staged runner is the canonical one now.
exec bash "$ROOT/scripts/run_staged_ablations.sh" "$@"

if [[ -z "${PY:-}" ]]; then
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PY="$ROOT/.venv/bin/python"
  elif [[ -x "$HOME/miniconda3/envs/dl_env/bin/python" ]]; then
    PY="$HOME/miniconda3/envs/dl_env/bin/python"
  elif [[ -x "$HOME/miniconda3/envs/dl/bin/python" ]]; then
    PY="$HOME/miniconda3/envs/dl/bin/python"
  else
    PY="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)"
  fi
fi

if [[ -z "$PY" ]]; then
  echo "[error] No Python interpreter found. Set PY=/path/to/python and retry." >&2
  exit 1
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT/.mplconfig}"
export TORCH_HOME="${TORCH_HOME:-$ROOT/data/torch}"
export HF_HOME="${HF_HOME:-$ROOT/data/hf}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export CHECKPOINT_TMPDIR="${CHECKPOINT_TMPDIR:-/tmp}"

SHARED_DATASET_DIR="${SHARED_DATASET_DIR:-/home/edxnG06/Maria_Siles/projecte-deep-learning-06/Shit_Learninig06/dataset}"
if [[ -d "$SHARED_DATASET_DIR/flickr30k_hf" ]]; then
  FLICKR30K_HF_CACHE="${FLICKR30K_HF_CACHE:-$SHARED_DATASET_DIR/flickr30k_hf}"
else
  FLICKR30K_HF_CACHE="${FLICKR30K_HF_CACHE:-$ROOT/data/flickr30k_hf}"
fi

if [[ -z "${FLICKR30K_PY:-}" ]]; then
  if "$PY" -c "import datasets" >/dev/null 2>&1; then
    FLICKR30K_PY="$PY"
  elif command -v python >/dev/null 2>&1 && python -c "import datasets" >/dev/null 2>&1; then
    FLICKR30K_PY="$(command -v python)"
  else
    FLICKR30K_PY="$PY"
  fi
fi

WANDB_PROJECT="${WANDB_PROJECT:-Clean}"
WANDB_ENTITY="${WANDB_ENTITY:-learning6}"
USE_WANDB="${USE_WANDB:-1}"

EPOCHS="${EPOCHS:-20}"
PATIENCE="${PATIENCE:-999}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-2}"
LOG_STEP="${LOG_STEP:-20}"
LR="${LR:-1e-3}"
BASE_LR="${BASE_LR:-1e-4}"
MAX_LR="${MAX_LR:-1e-3}"
STEP_SIZE_UP_EPOCHS="${STEP_SIZE_UP_EPOCHS:-4}"
VOCAB_THRESHOLD="${VOCAB_THRESHOLD:-5}"
EMBED_SIZE="${EMBED_SIZE:-256}"
HIDDEN_SIZE="${HIDDEN_SIZE:-512}"
ATTENTION_DIM="${ATTENTION_DIM:-256}"
NUM_LAYERS="${NUM_LAYERS:-1}"
DROPOUT="${DROPOUT:-0.5}"
DS_LAMBDA="${DS_LAMBDA:-1.0}"
SEMANTIC_TEMP="${SEMANTIC_TEMP:-10.0}"
CHECKPOINT_MODE="${CHECKPOINT_MODE:-light}"
DRY_RUN="${DRY_RUN:-0}"
MAX_RUNS="${MAX_RUNS:-0}"
RUN_ONLY="${RUN_ONLY:-}"

mkdir -p "$MPLCONFIGDIR" "$TORCH_HOME" "$HF_HOME" checkpoints checkpoints_attention data/embeddings data/flickr30k_hf data/coco2017 logs

RUN_COUNT=0

usage() {
  cat <<'EOF'
Usage: scripts/run_clean_ablations.sh <group>

Main groups:
  smoke                 Tiny local run to verify Linux/VM wiring.
  current               Recreate the useful runs already visible in wandb.
  backbone              ResNet50, ResNet152, EfficientNet-B0, VGG16.
  embeddings            Scratch, GloVe 50/100/200/300, Word2Vec if present.
  lstm                  Unidirectional vs bidirectional decoder.
  loss                  CE, CE + label smoothing, semantic CE with GloVe.
  dataset               Flickr8k, Flickr30k if prepared, COCO2017 if prepared.
  learning-rate         5e-4, 1e-3, 2e-3, CyclicLR.
  tuning                Batch, embedding, hidden size, vocab threshold sweep.
  implementation-first  Small set: efficientnet, cycliclr, bidir.
  all                   Runs the main comparison groups in order.

Utility groups:
  prepare-coco          Prepare data/coco2017 from /home/datasets/coco.
  list                  Show groups.

Useful env vars:
  PY=/path/to/python
  EPOCHS=20 BATCH_SIZE=32 NUM_WORKERS=2
  WANDB_PROJECT=Clean WANDB_ENTITY=learning6 WANDB_MODE=online|offline
  DRY_RUN=1             Print commands without executing them.
  MAX_RUNS=3            Stop after N runs, useful on the capped VM.
  RUN_ONLY=substring    Run only names containing the substring.
  MAX_TRAIN_IMAGES=6000 MAX_VAL_IMAGES=500 MAX_TEST_IMAGES=500
  NO_PRETRAINED_BACKBONE=1
  CHECKPOINT_MODE=light  Use full, light, or none.
  SHARED_DATASET_DIR=/path/to/dataset  Reuse shared GloVe/Flickr30k cache.
  FLICKR30K_PY=/path/to/python  Python with HuggingFace datasets installed.
EOF
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "[missing] $label: $path" >&2
    return 1
  fi
}

dataset_available() {
  local dataset="$1"
  case "$dataset" in
    flickr8k)
      [[ -d data/flickr8k/Images && -f data/flickr8k/captions.txt ]]
      ;;
    flickr30k)
      [[ -d "$FLICKR30K_HF_CACHE" ]]
      ;;
    coco2017)
      [[ -d data/coco2017/Images && -f data/coco2017/captions.txt ]]
      ;;
    *)
      return 1
      ;;
  esac
}

dataset_args() {
  local dataset="$1"
  case "$dataset" in
    flickr8k)
      DATASET_ARGS=(--images-dir data/flickr8k/Images --captions-csv data/flickr8k/captions.txt --vocab-path data/flickr8k/vocab.pkl)
      ;;
    flickr30k)
      DATASET_ARGS=(--flickr30k-hf --flickr30k-hf-cache "$FLICKR30K_HF_CACHE" --vocab-path data/flickr30k_hf/vocab.pkl)
      ;;
    coco2017)
      DATASET_ARGS=(--images-dir data/coco2017/Images --captions-csv data/coco2017/captions.txt --vocab-path data/coco2017/vocab.pkl)
      ;;
    *)
      echo "[error] Unknown dataset: $dataset" >&2
      exit 1
      ;;
  esac
}

slug() {
  local value="$1"
  value="${value//./p}"
  value="${value//-}"
  value="${value//_/-}"
  echo "$value"
}

run_shell() {
  local run_name="$1"
  shift

  if [[ -n "$RUN_ONLY" && "$run_name" != *"$RUN_ONLY"* ]]; then
    echo "[skip] $run_name does not match RUN_ONLY=$RUN_ONLY"
    return 0
  fi
  if (( MAX_RUNS > 0 && RUN_COUNT >= MAX_RUNS )); then
    echo "[stop] MAX_RUNS=$MAX_RUNS reached"
    return 0
  fi

  RUN_COUNT=$((RUN_COUNT + 1))
  echo
  echo "===== [$RUN_COUNT] $run_name ====="
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[dry-run]'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

run_train() {
  local arch="$1"
  local dataset="$2"
  local run_name="$3"
  local backbone="$4"
  shift 4

  if ! dataset_available "$dataset"; then
    echo "[skip] dataset not prepared: $dataset"
    return 0
  fi
  dataset_args "$dataset"
  local train_py="$PY"
  if [[ "$dataset" == "flickr30k" ]]; then
    train_py="$FLICKR30K_PY"
  fi

  local module checkpoint_root
  local arch_args=()
  if [[ "$arch" == "baseline" ]]; then
    module="src.baseline.train"
    checkpoint_root="checkpoints/$run_name"
    arch_args=(--num-layers "$NUM_LAYERS")
  elif [[ "$arch" == "attention" ]]; then
    module="src.attention.train"
    checkpoint_root="checkpoints_attention/$run_name"
    arch_args=(--attention-dim "$ATTENTION_DIM" --ds-lambda "$DS_LAMBDA")
    if [[ -n "${FINETUNE_CNN_EPOCH:-}" ]]; then
      arch_args+=(--finetune-cnn-epoch "$FINETUNE_CNN_EPOCH")
    fi
  else
    echo "[error] Unknown architecture: $arch" >&2
    exit 1
  fi

  local wandb_args=()
  if [[ "$USE_WANDB" == "1" ]]; then
    wandb_args=(--wandb --wandb-project "$WANDB_PROJECT")
    if [[ -n "$WANDB_ENTITY" ]]; then
      wandb_args+=(--wandb-entity "$WANDB_ENTITY")
    fi
  fi

  local limit_args=()
  if [[ -n "${MAX_TRAIN_IMAGES:-}" ]]; then
    limit_args+=(--max-train-images "$MAX_TRAIN_IMAGES")
  fi
  if [[ -n "${MAX_VAL_IMAGES:-}" ]]; then
    limit_args+=(--max-val-images "$MAX_VAL_IMAGES")
  fi
  if [[ -n "${MAX_TEST_IMAGES:-}" ]]; then
    limit_args+=(--max-test-images "$MAX_TEST_IMAGES")
  fi
  if [[ "${NO_PRETRAINED_BACKBONE:-0}" == "1" ]]; then
    limit_args+=(--no-pretrained-backbone)
  fi

  run_shell "$run_name" \
    env MPLCONFIGDIR="$MPLCONFIGDIR" TORCH_HOME="$TORCH_HOME" HF_HOME="$HF_HOME" PYTORCH_ENABLE_MPS_FALLBACK="$PYTORCH_ENABLE_MPS_FALLBACK" PYTHONDONTWRITEBYTECODE="$PYTHONDONTWRITEBYTECODE" CHECKPOINT_TMPDIR="$CHECKPOINT_TMPDIR" \
    "$train_py" -m "$module" \
    "${DATASET_ARGS[@]}" \
    --checkpoints-dir "$checkpoint_root" \
    --vocab-threshold "$VOCAB_THRESHOLD" \
    --embed-size "$EMBED_SIZE" \
    --hidden-size "$HIDDEN_SIZE" \
    --dropout "$DROPOUT" \
    --backbone "$backbone" \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --log-step "$LOG_STEP" \
    --semantic-temp "$SEMANTIC_TEMP" \
    --checkpoint-mode "$CHECKPOINT_MODE" \
    "${arch_args[@]}" \
    "${limit_args[@]}" \
    "${wandb_args[@]}" \
    --run-name "$run_name" \
    "$@"
}

run_exp() {
  local arch="$1"
  local dataset="$2"
  local backbone="$3"
  local embedding="$4"
  local loss="$5"
  local lr="$6"
  local scheduler="$7"
  local suffix="${8:-}"
  shift 8

  local name_lr="lr$(slug "$lr")"
  if [[ "$scheduler" == "cyclic" ]]; then
    name_lr="cycliclr"
  fi

  local run_name="$arch-$dataset-$backbone-$embedding-$loss-$name_lr-${EPOCHS}ep"
  if [[ -n "$suffix" ]]; then
    run_name="$run_name-$suffix"
  fi

  local extra=(--lr "$lr" --scheduler "$scheduler")
  if [[ "$scheduler" == "cyclic" ]]; then
    extra+=(--base-lr "$BASE_LR" --max-lr "$MAX_LR" --step-size-up-epochs "$STEP_SIZE_UP_EPOCHS")
  fi

  case "$embedding" in
    scratch)
      ;;
    glove50|glove100|glove200|glove300)
      local dim="${embedding#glove}"
      local glove_path="$ROOT/data/embeddings/glove.6B.${dim}d.txt"
      if ! require_file "$glove_path" "GloVe ${dim}d"; then
        echo "[skip] $run_name"
        return 0
      fi
      extra+=(--glove-path "$glove_path")
      ;;
    word2vec)
      local w2v_path="$ROOT/data/embeddings/GoogleNews-vectors-negative300.bin"
      if ! require_file "$w2v_path" "GoogleNews Word2Vec"; then
        echo "[skip] $run_name"
        return 0
      fi
      extra+=(--word2vec-path "$w2v_path" --word2vec-binary)
      ;;
    *)
      echo "[error] Unknown embedding: $embedding" >&2
      exit 1
      ;;
  esac

  case "$loss" in
    ce)
      extra+=(--label-smoothing 0.0)
      if [[ "$embedding" != "scratch" ]]; then
        extra+=(--no-semantic-loss)
      fi
      ;;
    ls01)
      extra+=(--label-smoothing 0.1)
      if [[ "$embedding" != "scratch" ]]; then
        extra+=(--no-semantic-loss)
      fi
      ;;
    semantic)
      if [[ "$embedding" == "scratch" ]]; then
        echo "[skip] semantic loss requires pretrained embeddings: $run_name"
        return 0
      fi
      ;;
    *)
      echo "[error] Unknown loss: $loss" >&2
      exit 1
      ;;
  esac

  run_train "$arch" "$dataset" "$run_name" "$backbone" "${extra[@]}" "$@"
}

run_pair() {
  run_exp baseline "$@"
  run_exp attention "$@"
}

run_smoke() {
  local old_epochs="$EPOCHS"
  EPOCHS="${SMOKE_EPOCHS:-1}"
  run_exp baseline flickr8k resnet50 scratch ce "$LR" plateau smoke --max-train-images 64 --max-val-images 32 --max-test-images 32 --skip-test-captioning
  run_exp attention flickr8k resnet50 scratch ce "$LR" plateau smoke --max-train-images 64 --max-val-images 32 --max-test-images 32 --skip-test-captioning
  EPOCHS="$old_epochs"
}

run_current() {
  run_exp baseline flickr8k resnet50 scratch ce 1e-3 plateau ""
  run_exp baseline flickr8k resnet152 scratch ce 1e-3 plateau ""
  run_exp attention flickr8k resnet50 scratch ce 1e-3 plateau ""
  run_exp baseline flickr8k resnet50 glove50 ce 1e-3 plateau ""
  run_exp baseline flickr8k resnet50 glove200 ce 1e-3 plateau ""
  run_exp baseline flickr8k resnet50 glove300 ce 1e-3 plateau ""
  run_exp attention flickr8k resnet50 glove300 ce 1e-3 plateau ""
  run_exp baseline flickr30k resnet50 scratch ce 1e-3 plateau ""
}

run_backbone() {
  for backbone in resnet50 resnet152 efficientnet_b0 vgg16; do
    run_pair flickr8k "$backbone" scratch ce 1e-3 plateau ""
  done
}

run_embeddings() {
  for embedding in scratch glove50 glove100 glove200 glove300 word2vec; do
    run_pair flickr8k resnet50 "$embedding" ce 1e-3 plateau ""
  done
}

run_lstm() {
  run_pair flickr8k resnet50 scratch ce 1e-3 plateau uni --decoder-direction uni
  run_pair flickr8k resnet50 scratch ce 1e-3 plateau bidir --decoder-direction bidir --skip-test-captioning
}

run_loss() {
  run_pair flickr8k resnet50 scratch ce 1e-3 plateau ""
  run_pair flickr8k resnet50 scratch ls01 1e-3 plateau ""
  run_pair flickr8k resnet50 glove300 semantic 1e-3 plateau ""
}

run_dataset() {
  run_pair flickr8k resnet50 scratch ce 1e-3 plateau ""
  run_pair flickr30k resnet50 scratch ce 1e-3 plateau ""
  run_pair coco2017 resnet50 scratch ce 1e-3 plateau ""
}

run_learning_rate() {
  for lr in 5e-4 1e-3 2e-3; do
    run_pair flickr8k resnet50 scratch ce "$lr" plateau ""
  done
  run_pair flickr8k resnet50 scratch ce "$LR" cyclic ""
}

run_tuning() {
  for batch in 16 32 64; do
    run_pair flickr8k resnet50 scratch ce 1e-3 plateau "batch$batch" --batch-size "$batch"
  done
  for embed in 128 256 512; do
    run_pair flickr8k resnet50 scratch ce 1e-3 plateau "emb$embed" --embed-size "$embed"
  done
  for hidden in 256 512 1024; do
    run_pair flickr8k resnet50 scratch ce 1e-3 plateau "hid$hidden" --hidden-size "$hidden"
  done
  for threshold in 3 5; do
    run_pair flickr8k resnet50 scratch ce 1e-3 plateau "vocab$threshold" --vocab-threshold "$threshold"
  done
  for layers in 2 3; do
    run_exp baseline flickr8k resnet50 scratch ce 1e-3 plateau "layers$layers" --num-layers "$layers"
  done
  for att_dim in 128 512; do
    run_exp attention flickr8k resnet50 scratch ce 1e-3 plateau "att$att_dim" --attention-dim "$att_dim"
  done
}

run_implementation_first() {
  run_pair flickr8k efficientnet_b0 scratch ce 1e-3 plateau ""
  run_pair flickr8k resnet50 scratch ce "$LR" cyclic ""
  run_pair flickr8k resnet50 scratch ce 1e-3 plateau bidir --decoder-direction bidir --skip-test-captioning
}

run_all() {
  run_current
  run_backbone
  run_embeddings
  run_lstm
  run_loss
  run_dataset
  run_learning_rate
}

case "${1:-}" in
  smoke) run_smoke ;;
  current) run_current ;;
  backbone) run_backbone ;;
  embeddings) run_embeddings ;;
  lstm) run_lstm ;;
  loss) run_loss ;;
  dataset) run_dataset ;;
  learning-rate) run_learning_rate ;;
  tuning) run_tuning ;;
  implementation-first) run_implementation_first ;;
  all) run_all ;;
  prepare-coco) "$PY" scripts/prepare_coco2017.py --source-root "${COCO_SOURCE_ROOT:-/home/datasets/coco}" --root data/coco2017 ;;
  list) usage ;;
  *) usage; exit 1 ;;
esac

echo
echo "[done] scheduled/executed runs: $RUN_COUNT"
