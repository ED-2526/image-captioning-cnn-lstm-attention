#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

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
if [[ -z "${TORCH_HOME:-}" || "${TORCH_HOME:-}" == "$ROOT/data/torch" || "${TORCH_HOME:-}" == "$ROOT/data/torch/"* ]]; then
  export TORCH_HOME="/tmp/torch-${USER:-user}"
else
  export TORCH_HOME
fi
export HF_HOME="${HF_HOME:-$ROOT/data/hf}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export CHECKPOINT_TMPDIR="${CHECKPOINT_TMPDIR:-/tmp}"
export WANDB_DIR="${WANDB_DIR:-/tmp/wandb-${USER:-user}}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/wandb-cache-${USER:-user}}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/tmp/wandb-config-${USER:-user}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg-cache-${USER:-user}}"

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

WANDB_PROJECT="${WANDB_PROJECT:-altre_net}"
WANDB_ENTITY="${WANDB_ENTITY:-learning6}"
USE_WANDB="${USE_WANDB:-1}"
WANDB_RICH="${WANDB_RICH:-0}"

PARALLEL_RUNS="${PARALLEL_RUNS:-1}"
PARALLEL_LOG_DIR="${PARALLEL_LOG_DIR:-/tmp/projecte-deep-learning-06-parallel-${USER:-user}}"
EPOCHS="${EPOCHS:-20}"
PATIENCE="${PATIENCE:-999}"
BATCH_SIZE="${BATCH_SIZE:-32}"
if [[ -z "${NUM_WORKERS:-}" ]]; then
  if (( PARALLEL_RUNS > 1 )); then
    NUM_WORKERS=1
  else
    NUM_WORKERS=2
  fi
fi
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
CHECKPOINT_MODE="${CHECKPOINT_MODE:-none}"
SAVE_EVERY_EPOCH="${SAVE_EVERY_EPOCH:-0}"
SKIP_LOCAL_ARTIFACTS="${SKIP_LOCAL_ARTIFACTS:-1}"
DRY_RUN="${DRY_RUN:-0}"
MAX_RUNS="${MAX_RUNS:-0}"
RUN_ONLY="${RUN_ONLY:-}"

BASE_DATASET="${BASE_DATASET:-flickr8k}"
BEST_BASELINE_BACKBONE="${BEST_BASELINE_BACKBONE:-resnet50}"
BEST_ATTENTION_BACKBONE="${BEST_ATTENTION_BACKBONE:-resnet50}"
BEST_BASELINE_EMBEDDING="${BEST_BASELINE_EMBEDDING:-scratch}"
BEST_ATTENTION_EMBEDDING="${BEST_ATTENTION_EMBEDDING:-scratch}"
BEST_BASELINE_LSTM="${BEST_BASELINE_LSTM:-uni}"
BEST_ATTENTION_LSTM="${BEST_ATTENTION_LSTM:-uni}"
BEST_BASELINE_LOSS="${BEST_BASELINE_LOSS:-ce}"
BEST_ATTENTION_LOSS="${BEST_ATTENTION_LOSS:-ce}"
BEST_BASELINE_LR="${BEST_BASELINE_LR:-1e-3}"
BEST_ATTENTION_LR="${BEST_ATTENTION_LR:-1e-3}"

GLOVE_300="${GLOVE_300:-$ROOT/data/embeddings/glove.6B.300d.txt}"
WORD2VEC="${WORD2VEC:-$ROOT/data/embeddings/GoogleNews-vectors-negative300.bin}"

mkdir -p "$MPLCONFIGDIR" "$TORCH_HOME" "$HF_HOME" checkpoints checkpoints_attention data/embeddings data/flickr30k_hf data/coco2017 "$PARALLEL_LOG_DIR" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$XDG_CACHE_HOME"

RUN_COUNT=0
JOB_FAILURES=0
ACTIVE_PIDS=()
ACTIVE_RUNS=()

cleanup_parallel_runs() {
  if (( ${#ACTIVE_PIDS[@]} > 0 )); then
    echo
    echo "[parallel] stopping ${#ACTIVE_PIDS[@]} active run(s)..."
    kill -TERM "${ACTIVE_PIDS[@]}" 2>/dev/null || true
    sleep 2
    kill -KILL "${ACTIVE_PIDS[@]}" 2>/dev/null || true
  fi
}

on_interrupt() {
  cleanup_parallel_runs
  exit 130
}

trap on_interrupt INT TERM

usage() {
  cat <<'EOF'
Usage: scripts/run_clean_ablations.sh <stage>

Stages:
  smoke              1 tiny baseline + 1 tiny attention run.
  stage-backbone    Compare ResNet50, ResNet152, EfficientNet-B0, VGG16.
  stage-embeddings  Compare scratch, GloVe300, Word2Vec using BEST_*_BACKBONE.
  stage-lstm        Compare uni vs bidir using chosen backbone/embedding.
  stage-loss        Compare CE, label smoothing, and semantic loss when valid.
  stage-dataset     Compare Flickr8k, Flickr30k, COCO using chosen config.
  stage-lr          Compare 5e-4, 1e-3, 2e-3, and CyclicLR.
  stage-tuning      Small final sweep around the chosen config.
  all               Run stages with the BEST_* defaults currently set.

Aliases:
  backbone, embeddings, lstm, loss, dataset, learning-rate, tuning

Useful env vars:
  WANDB_PROJECT=altre_net WANDB_ENTITY=learning6 USE_WANDB=1 WANDB_RICH=0 WANDB_DIR=/tmp/wandb-$USER WANDB_CACHE_DIR=/tmp/wandb-cache-$USER WANDB_CONFIG_DIR=/tmp/wandb-config-$USER TORCH_HOME=/tmp/torch-$USER
  DRY_RUN=1 MAX_RUNS=3 RUN_ONLY=substring PARALLEL_RUNS=3
  EPOCHS=20 BATCH_SIZE=32 NUM_WORKERS=2 CHECKPOINT_MODE=none SAVE_EVERY_EPOCH=0 SKIP_LOCAL_ARTIFACTS=1
  MAX_TRAIN_IMAGES=6000 MAX_VAL_IMAGES=500 MAX_TEST_IMAGES=500
  BEST_BASELINE_BACKBONE=resnet50 BEST_ATTENTION_BACKBONE=resnet50
  BEST_BASELINE_EMBEDDING=scratch BEST_ATTENTION_EMBEDDING=scratch
  BEST_BASELINE_LSTM=uni BEST_ATTENTION_LSTM=uni
  BEST_BASELINE_LOSS=ce BEST_ATTENTION_LOSS=ce
  BEST_BASELINE_LR=1e-3 BEST_ATTENTION_LR=1e-3
EOF
}

dataset_available() {
  local dataset="$1"
  case "$dataset" in
    flickr8k) [[ -d data/flickr8k/Images && -f data/flickr8k/captions.txt ]] ;;
    flickr30k) [[ -d "$FLICKR30K_HF_CACHE" ]] ;;
    coco2017) [[ -d data/coco2017/Images && -f data/coco2017/captions.txt ]] ;;
    *) return 1 ;;
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
  elif (( PARALLEL_RUNS > 1 )); then
    while (( ${#ACTIVE_PIDS[@]} >= PARALLEL_RUNS )); do
      wait_for_oldest_run
    done
    local log_path="$PARALLEL_LOG_DIR/$run_name.log"
    echo "[parallel] starting in background; log=$log_path"
    "$@" > "$log_path" 2>&1 &
    ACTIVE_PIDS+=("$!")
    ACTIVE_RUNS+=("$run_name")
  else
    "$@"
  fi
}

wait_for_oldest_run() {
  local pid="${ACTIVE_PIDS[0]}"
  local run_name="${ACTIVE_RUNS[0]}"

  if wait "$pid"; then
    echo "[parallel] completed: $run_name"
  else
    local status=$?
    echo "[parallel] failed ($status): $run_name" >&2
    JOB_FAILURES=$((JOB_FAILURES + 1))
  fi

  ACTIVE_PIDS=("${ACTIVE_PIDS[@]:1}")
  ACTIVE_RUNS=("${ACTIVE_RUNS[@]:1}")
}

wait_for_parallel_runs() {
  if (( PARALLEL_RUNS <= 1 || DRY_RUN == 1 )); then
    return 0
  fi

  echo
  echo "[parallel] waiting for remaining runs..."
  while (( ${#ACTIVE_PIDS[@]} > 0 )); do
    wait_for_oldest_run
  done

  if (( JOB_FAILURES > 0 )); then
    echo "[error] $JOB_FAILURES parallel run(s) failed. Check $PARALLEL_LOG_DIR/*.log" >&2
    exit 1
  fi
}

embedding_args() {
  local embedding="$1"
  EMBEDDING_ARGS=()
  case "$embedding" in
    scratch)
      ;;
    glove300)
      if [[ ! -f "$GLOVE_300" ]]; then
        return 1
      fi
      EMBEDDING_ARGS=(--glove-path "$GLOVE_300")
      ;;
    word2vec)
      if [[ ! -f "$WORD2VEC" ]]; then
        return 1
      fi
      EMBEDDING_ARGS=(--word2vec-path "$WORD2VEC" --word2vec-binary)
      ;;
    *)
      echo "[error] Unknown embedding: $embedding" >&2
      exit 1
      ;;
  esac
}

loss_args() {
  local embedding="$1"
  local loss="$2"
  LOSS_ARGS=()
  case "$loss" in
    ce)
      LOSS_ARGS=(--label-smoothing 0.0)
      if [[ "$embedding" != "scratch" ]]; then
        LOSS_ARGS+=(--no-semantic-loss)
      fi
      ;;
    ls01)
      LOSS_ARGS=(--label-smoothing 0.1)
      if [[ "$embedding" != "scratch" ]]; then
        LOSS_ARGS+=(--no-semantic-loss)
      fi
      ;;
    semantic)
      if [[ "$embedding" == "scratch" ]]; then
        return 1
      fi
      LOSS_ARGS=()
      ;;
    *)
      echo "[error] Unknown loss: $loss" >&2
      exit 1
      ;;
  esac
}

run_train() {
  local stage="$1"
  local arch="$2"
  local dataset="$3"
  local backbone="$4"
  local embedding="$5"
  local lstm="$6"
  local loss="$7"
  local lr="$8"
  local scheduler="$9"
  local suffix="${10:-}"
  shift 10

  if ! dataset_available "$dataset"; then
    echo "[skip] dataset not prepared: $dataset"
    return 0
  fi
  dataset_args "$dataset"

  local train_py="$PY"
  if [[ "$dataset" == "flickr30k" ]]; then
    train_py="$FLICKR30K_PY"
  fi

  if ! embedding_args "$embedding"; then
    echo "[skip] missing files for embedding=$embedding"
    return 0
  fi
  if ! loss_args "$embedding" "$loss"; then
    echo "[skip] semantic loss requires pretrained embeddings"
    return 0
  fi

  local module checkpoint_root
  local arch_args=()
  if [[ "$arch" == "baseline" ]]; then
    module="src.baseline.train"
    checkpoint_root="checkpoints"
    arch_args=(--num-layers "$NUM_LAYERS")
  elif [[ "$arch" == "attention" ]]; then
    module="src.attention.train"
    checkpoint_root="checkpoints_attention"
    arch_args=(--attention-dim "$ATTENTION_DIM" --ds-lambda "$DS_LAMBDA")
    if [[ -n "${FINETUNE_CNN_EPOCH:-}" ]]; then
      arch_args+=(--finetune-cnn-epoch "$FINETUNE_CNN_EPOCH")
    fi
  else
    echo "[error] Unknown architecture: $arch" >&2
    exit 1
  fi

  local name_lr="lr$(slug "$lr")"
  local lr_args=(--lr "$lr" --scheduler "$scheduler")
  if [[ "$scheduler" == "cyclic" ]]; then
    name_lr="cycliclr"
    lr_args+=(--base-lr "$BASE_LR" --max-lr "$MAX_LR" --step-size-up-epochs "$STEP_SIZE_UP_EPOCHS")
  fi

  local run_name="$stage-$arch-$dataset-$backbone-$embedding-$lstm-$loss-$name_lr-${EPOCHS}ep"
  if [[ -n "$suffix" ]]; then
    run_name="$run_name-$suffix"
  fi

  local wandb_args=()
  if [[ "$USE_WANDB" == "1" ]]; then
    wandb_args=(
      --wandb
      --wandb-project "$WANDB_PROJECT"
      --wandb-group "$stage"
      --wandb-job-type "$arch"
      --wandb-tags "$stage,$arch,$dataset,$backbone,$embedding,$lstm,$loss"
    )
    if [[ -n "$WANDB_ENTITY" ]]; then
      wandb_args+=(--wandb-entity "$WANDB_ENTITY")
    fi
    if [[ "$WANDB_RICH" == "1" ]]; then
      wandb_args+=(--wandb-rich)
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

  local checkpoint_args=(--checkpoint-mode "$CHECKPOINT_MODE")
  if [[ "$SAVE_EVERY_EPOCH" == "1" ]]; then
    checkpoint_args+=(--save-every-epoch)
  fi
  if [[ "$SKIP_LOCAL_ARTIFACTS" == "1" ]]; then
    checkpoint_args+=(--skip-local-artifacts)
  fi

  run_shell "$run_name" \
    env MPLCONFIGDIR="$MPLCONFIGDIR" TORCH_HOME="$TORCH_HOME" HF_HOME="$HF_HOME" WANDB_DIR="$WANDB_DIR" WANDB_CACHE_DIR="$WANDB_CACHE_DIR" WANDB_CONFIG_DIR="$WANDB_CONFIG_DIR" XDG_CACHE_HOME="$XDG_CACHE_HOME" PYTORCH_ENABLE_MPS_FALLBACK="$PYTORCH_ENABLE_MPS_FALLBACK" PYTHONDONTWRITEBYTECODE="$PYTHONDONTWRITEBYTECODE" CHECKPOINT_TMPDIR="$CHECKPOINT_TMPDIR" \
    "$train_py" -m "$module" \
    "${DATASET_ARGS[@]}" \
    --checkpoints-dir "$checkpoint_root/$run_name" \
    --vocab-threshold "$VOCAB_THRESHOLD" \
    --embed-size "$EMBED_SIZE" \
    --hidden-size "$HIDDEN_SIZE" \
    --dropout "$DROPOUT" \
    --backbone "$backbone" \
    --decoder-direction "$lstm" \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --log-step "$LOG_STEP" \
    --semantic-temp "$SEMANTIC_TEMP" \
    "${checkpoint_args[@]}" \
    "${arch_args[@]}" \
    "${EMBEDDING_ARGS[@]}" \
    "${LOSS_ARGS[@]}" \
    "${lr_args[@]}" \
    "${limit_args[@]}" \
    "${wandb_args[@]}" \
    --run-name "$run_name" \
    "$@"
}

run_arch() {
  local stage="$1"
  local arch="$2"
  local dataset="$3"
  local backbone="$4"
  local embedding="$5"
  local lstm="$6"
  local loss="$7"
  local lr="$8"
  local scheduler="$9"
  local suffix="${10:-}"
  shift 10
  run_train "$stage" "$arch" "$dataset" "$backbone" "$embedding" "$lstm" "$loss" "$lr" "$scheduler" "$suffix" "$@"
}

run_pair() {
  local stage="$1"
  local dataset="$2"
  local baseline_backbone="$3"
  local attention_backbone="$4"
  local baseline_embedding="$5"
  local attention_embedding="$6"
  local baseline_lstm="$7"
  local attention_lstm="$8"
  local baseline_loss="$9"
  local attention_loss="${10}"
  local baseline_lr="${11}"
  local attention_lr="${12}"
  local scheduler="${13}"
  local suffix="${14:-}"
  shift 14
  run_arch "$stage" baseline "$dataset" "$baseline_backbone" "$baseline_embedding" "$baseline_lstm" "$baseline_loss" "$baseline_lr" "$scheduler" "$suffix" "$@"
  run_arch "$stage" attention "$dataset" "$attention_backbone" "$attention_embedding" "$attention_lstm" "$attention_loss" "$attention_lr" "$scheduler" "$suffix" "$@"
}

run_smoke() {
  local old_epochs="$EPOCHS"
  EPOCHS="${SMOKE_EPOCHS:-1}"
  run_arch smoke baseline flickr8k resnet50 scratch uni ce 1e-3 plateau smoke --max-train-images 64 --max-val-images 32 --max-test-images 32 --skip-test-captioning --checkpoint-mode none
  run_arch smoke attention flickr8k resnet50 scratch uni ce 1e-3 plateau smoke --max-train-images 64 --max-val-images 32 --max-test-images 32 --skip-test-captioning --checkpoint-mode none
  EPOCHS="$old_epochs"
}

run_stage_backbone() {
  for backbone in resnet50 resnet152 efficientnet_b0 vgg16; do
    run_pair stage-backbone "$BASE_DATASET" "$backbone" "$backbone" scratch scratch uni uni ce ce 1e-3 1e-3 plateau ""
  done
}

run_stage_embeddings() {
  for embedding in scratch glove300 word2vec; do
    run_pair stage-embeddings "$BASE_DATASET" "$BEST_BASELINE_BACKBONE" "$BEST_ATTENTION_BACKBONE" "$embedding" "$embedding" uni uni ce ce 1e-3 1e-3 plateau ""
  done
}

run_stage_lstm() {
  for direction in uni bidir; do
    local extra=()
    if [[ "$direction" == "bidir" ]]; then
      extra=(--skip-test-captioning)
    fi
    run_pair stage-lstm "$BASE_DATASET" "$BEST_BASELINE_BACKBONE" "$BEST_ATTENTION_BACKBONE" "$BEST_BASELINE_EMBEDDING" "$BEST_ATTENTION_EMBEDDING" "$direction" "$direction" ce ce 1e-3 1e-3 plateau "$direction" "${extra[@]}"
  done
}

run_stage_loss() {
  for loss in ce ls01 semantic; do
    run_pair stage-loss "$BASE_DATASET" "$BEST_BASELINE_BACKBONE" "$BEST_ATTENTION_BACKBONE" "$BEST_BASELINE_EMBEDDING" "$BEST_ATTENTION_EMBEDDING" "$BEST_BASELINE_LSTM" "$BEST_ATTENTION_LSTM" "$loss" "$loss" 1e-3 1e-3 plateau "$loss"
  done
}

run_stage_dataset() {
  for dataset in flickr8k flickr30k coco2017; do
    run_pair stage-dataset "$dataset" "$BEST_BASELINE_BACKBONE" "$BEST_ATTENTION_BACKBONE" "$BEST_BASELINE_EMBEDDING" "$BEST_ATTENTION_EMBEDDING" "$BEST_BASELINE_LSTM" "$BEST_ATTENTION_LSTM" "$BEST_BASELINE_LOSS" "$BEST_ATTENTION_LOSS" 1e-3 1e-3 plateau ""
  done
}

run_stage_lr() {
  for lr in 5e-4 1e-3 2e-3; do
    run_pair stage-lr "$BASE_DATASET" "$BEST_BASELINE_BACKBONE" "$BEST_ATTENTION_BACKBONE" "$BEST_BASELINE_EMBEDDING" "$BEST_ATTENTION_EMBEDDING" "$BEST_BASELINE_LSTM" "$BEST_ATTENTION_LSTM" "$BEST_BASELINE_LOSS" "$BEST_ATTENTION_LOSS" "$lr" "$lr" plateau ""
  done
  run_pair stage-lr "$BASE_DATASET" "$BEST_BASELINE_BACKBONE" "$BEST_ATTENTION_BACKBONE" "$BEST_BASELINE_EMBEDDING" "$BEST_ATTENTION_EMBEDDING" "$BEST_BASELINE_LSTM" "$BEST_ATTENTION_LSTM" "$BEST_BASELINE_LOSS" "$BEST_ATTENTION_LOSS" "$LR" "$LR" cyclic ""
}

run_stage_tuning() {
  for batch in 16 32 64; do
    run_pair stage-tuning "$BASE_DATASET" "$BEST_BASELINE_BACKBONE" "$BEST_ATTENTION_BACKBONE" "$BEST_BASELINE_EMBEDDING" "$BEST_ATTENTION_EMBEDDING" "$BEST_BASELINE_LSTM" "$BEST_ATTENTION_LSTM" "$BEST_BASELINE_LOSS" "$BEST_ATTENTION_LOSS" "$BEST_BASELINE_LR" "$BEST_ATTENTION_LR" plateau "batch$batch" --batch-size "$batch"
  done
  for embed in 128 256 512; do
    run_pair stage-tuning "$BASE_DATASET" "$BEST_BASELINE_BACKBONE" "$BEST_ATTENTION_BACKBONE" scratch scratch "$BEST_BASELINE_LSTM" "$BEST_ATTENTION_LSTM" "$BEST_BASELINE_LOSS" "$BEST_ATTENTION_LOSS" "$BEST_BASELINE_LR" "$BEST_ATTENTION_LR" plateau "emb$embed" --embed-size "$embed"
  done
  for hidden in 256 512 1024; do
    run_pair stage-tuning "$BASE_DATASET" "$BEST_BASELINE_BACKBONE" "$BEST_ATTENTION_BACKBONE" "$BEST_BASELINE_EMBEDDING" "$BEST_ATTENTION_EMBEDDING" "$BEST_BASELINE_LSTM" "$BEST_ATTENTION_LSTM" "$BEST_BASELINE_LOSS" "$BEST_ATTENTION_LOSS" "$BEST_BASELINE_LR" "$BEST_ATTENTION_LR" plateau "hid$hidden" --hidden-size "$hidden"
  done
  for threshold in 3 5; do
    run_pair stage-tuning "$BASE_DATASET" "$BEST_BASELINE_BACKBONE" "$BEST_ATTENTION_BACKBONE" "$BEST_BASELINE_EMBEDDING" "$BEST_ATTENTION_EMBEDDING" "$BEST_BASELINE_LSTM" "$BEST_ATTENTION_LSTM" "$BEST_BASELINE_LOSS" "$BEST_ATTENTION_LOSS" "$BEST_BASELINE_LR" "$BEST_ATTENTION_LR" plateau "vocab$threshold" --vocab-threshold "$threshold"
  done
  for layers in 2 3; do
    run_arch stage-tuning baseline "$BASE_DATASET" "$BEST_BASELINE_BACKBONE" "$BEST_BASELINE_EMBEDDING" "$BEST_BASELINE_LSTM" "$BEST_BASELINE_LOSS" "$BEST_BASELINE_LR" plateau "layers$layers" --num-layers "$layers"
  done
  for att_dim in 128 512; do
    run_arch stage-tuning attention "$BASE_DATASET" "$BEST_ATTENTION_BACKBONE" "$BEST_ATTENTION_EMBEDDING" "$BEST_ATTENTION_LSTM" "$BEST_ATTENTION_LOSS" "$BEST_ATTENTION_LR" plateau "att$att_dim" --attention-dim "$att_dim"
  done
}

run_all() {
  run_stage_backbone
  run_stage_embeddings
  run_stage_lstm
  run_stage_loss
  run_stage_dataset
  run_stage_lr
}

case "${1:-}" in
  smoke) run_smoke ;;
  stage-backbone|backbone) run_stage_backbone ;;
  stage-embeddings|embeddings) run_stage_embeddings ;;
  stage-lstm|lstm) run_stage_lstm ;;
  stage-loss|loss) run_stage_loss ;;
  stage-dataset|dataset) run_stage_dataset ;;
  stage-lr|learning-rate|lr) run_stage_lr ;;
  stage-tuning|tuning) run_stage_tuning ;;
  all) run_all ;;
  prepare-coco) "$PY" scripts/prepare_coco2017.py --source-root "${COCO_SOURCE_ROOT:-/home/datasets/coco}" --root data/coco2017 ;;
  list|-h|--help|"") usage ;;
  *) usage; exit 1 ;;
esac

wait_for_parallel_runs

echo
echo "[done] scheduled/executed runs: $RUN_COUNT"
