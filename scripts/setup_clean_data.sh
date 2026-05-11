#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_FOR_PREP="${PYTHON_FOR_PREP:-${PY:-python}}"
KAGGLE_BIN="${KAGGLE_BIN:-$(command -v kaggle 2>/dev/null || true)}"
if [[ -z "$KAGGLE_BIN" && -x "$HOME/miniconda3/bin/kaggle" ]]; then
  KAGGLE_BIN="$HOME/miniconda3/bin/kaggle"
fi

mkdir -p data/embeddings data/flickr30k data/flickr30k_hf data/coco2017 data/torch logs

SHARED_DATASET_DIR="${SHARED_DATASET_DIR:-/home/edxnG06/Maria_Siles/projecte-deep-learning-06/Shit_Learninig06/dataset}"

link_shared_file() {
  local source="$1"
  local target="$2"
  if [[ -f "$source" && ! -e "$target" ]]; then
    ln -s "$source" "$target"
    echo "[link] $target -> $source"
  fi
}

link_shared_assets() {
  if [[ ! -d "$SHARED_DATASET_DIR" ]]; then
    return 0
  fi
  for dim in 50 100 200 300; do
    link_shared_file "$SHARED_DATASET_DIR/glove.6B.${dim}d.txt" "data/embeddings/glove.6B.${dim}d.txt"
  done
  link_shared_file "$SHARED_DATASET_DIR/vocab_flickr30k.pkl" "data/flickr30k_hf/vocab.pkl"
}

extract_zip_noclobber() {
  "$PYTHON_FOR_PREP" - "$1" "$2" <<'PY'
from pathlib import Path
from zipfile import ZipFile
import sys

zip_path = Path(sys.argv[1])
dest = Path(sys.argv[2])
dest.mkdir(parents=True, exist_ok=True)
with ZipFile(zip_path) as zf:
    for info in zf.infolist():
        target = dest / info.filename
        if target.exists():
            continue
        zf.extract(info, dest)
PY
}

extract_glove_dims() {
  "$PYTHON_FOR_PREP" - "$1" "$2" "${@:3}" <<'PY'
from pathlib import Path
from zipfile import ZipFile
import shutil
import sys

zip_path = Path(sys.argv[1])
dest = Path(sys.argv[2])
dims = sys.argv[3:]
dest.mkdir(parents=True, exist_ok=True)
with ZipFile(zip_path) as zf:
    for dim in dims:
        member = f"glove.6B.{dim}d.txt"
        target = dest / member
        if target.exists():
            continue
        with zf.open(member) as src, target.open("wb") as out:
            shutil.copyfileobj(src, out)
PY
}

download_embeddings() {
  link_shared_assets

  local glove_zip="data/embeddings/glove.6B.zip"
  local w2v_gz="data/embeddings/GoogleNews-vectors-negative300.bin.gz"
  local w2v_bin="data/embeddings/GoogleNews-vectors-negative300.bin"
  local glove_dims="${GLOVE_DIMS:-50 100 200 300}"

  local missing_glove=0
  for dim in $glove_dims; do
    if [[ ! -f "data/embeddings/glove.6B.${dim}d.txt" ]]; then
      missing_glove=1
    fi
  done

  if [[ "$missing_glove" == "1" ]]; then
    local glove_ready=0
    if [[ -f "$glove_zip" ]]; then
      echo "[embeddings] trying existing GloVe zip: $glove_zip"
      if extract_glove_dims "$glove_zip" data/embeddings $glove_dims; then
        glove_ready=1
      else
        echo "[embeddings] existing GloVe zip is incomplete; resuming download..."
      fi
    fi
    if [[ "$glove_ready" == "0" ]]; then
      echo "[embeddings] downloading GloVe 6B..."
      curl -fL -C - -o "$glove_zip" https://nlp.stanford.edu/data/glove.6B.zip
      extract_glove_dims "$glove_zip" data/embeddings $glove_dims
    fi
  else
    echo "[embeddings] GloVe already present for dims: $glove_dims"
  fi

  if [[ "${SKIP_WORD2VEC:-1}" == "1" ]]; then
    echo "[embeddings] skipping Word2Vec by default; set SKIP_WORD2VEC=0 to download it"
  elif [[ ! -f "$w2v_bin" ]]; then
    echo "[embeddings] downloading GoogleNews Word2Vec..."
    curl -fL -C - -o "$w2v_gz" \
      https://huggingface.co/LoganKilpatrick/GoogleNews-vectors-negative300/resolve/main/GoogleNews-vectors-negative300.bin.gz
    gunzip -k -f "$w2v_gz"
  else
    echo "[embeddings] Word2Vec already present: $w2v_bin"
  fi
}

download_flickr30k() {
  if [[ -d "$SHARED_DATASET_DIR/flickr30k_hf" ]]; then
    echo "[flickr30k] using shared HuggingFace cache: $SHARED_DATASET_DIR/flickr30k_hf"
    return 0
  fi

  if [[ -f data/flickr30k_hf/captions.txt && -d data/flickr30k_hf/Images ]]; then
    echo "[flickr30k] already prepared under data/flickr30k_hf"
    return 0
  fi

  if [[ ! -f "$HOME/.kaggle/kaggle.json" ]]; then
    cat <<EOF
[flickr30k] Missing Kaggle credentials at $HOME/.kaggle/kaggle.json
            Put your kaggle.json there, then rerun:
              mkdir -p ~/.kaggle
              chmod 600 ~/.kaggle/kaggle.json
              bash scripts/setup_clean_data.sh
EOF
    return 1
  fi

  if [[ -z "$KAGGLE_BIN" ]]; then
    echo "[flickr30k] Kaggle CLI not found. Install it or set KAGGLE_BIN=/path/to/kaggle."
    return 1
  fi

  local kaggle_dataset="${FLICKR30K_KAGGLE_DATASET:-eeshawn/flickr30k}"
  local flickr_zip
  flickr_zip="$(find data/flickr30k -maxdepth 1 -name '*.zip' -print -quit)"

  if [[ -z "$flickr_zip" ]]; then
    echo "[flickr30k] downloading Kaggle dataset: $kaggle_dataset"
    "$KAGGLE_BIN" datasets download -d "$kaggle_dataset" -p data/flickr30k
    flickr_zip="$(find data/flickr30k -maxdepth 1 -name '*.zip' -print -quit)"
  else
    echo "[flickr30k] zip already present: $flickr_zip"
  fi

  if [[ -z "$flickr_zip" ]]; then
    echo "[flickr30k] Kaggle did not leave a zip in data/flickr30k" >&2
    return 1
  fi

  extract_zip_noclobber "$flickr_zip" data/flickr30k
  "$PYTHON_FOR_PREP" scripts/prepare_flickr30k_kaggle.py --root data/flickr30k --out data/flickr30k_hf
}

download_coco2017() {
  local root="data/coco2017"
  local source_root="${COCO_SOURCE_ROOT:-/home/datasets/coco}"
  mkdir -p "$root"

  if [[ -d "$source_root/train2017" && -d "$source_root/val2017" && -f "$source_root/annotations/captions_train2017.json" ]]; then
    echo "[coco] using shared COCO source: $source_root"
    "$PYTHON_FOR_PREP" scripts/prepare_coco2017.py --source-root "$source_root" --root "$root"
    return 0
  fi

  if [[ ! -f "$root/train2017.zip" ]]; then
    echo "[coco] downloading train2017..."
    curl -fL -C - -o "$root/train2017.zip" http://images.cocodataset.org/zips/train2017.zip
  else
    echo "[coco] train2017.zip already present"
  fi

  if [[ ! -f "$root/val2017.zip" ]]; then
    echo "[coco] downloading val2017..."
    curl -fL -C - -o "$root/val2017.zip" http://images.cocodataset.org/zips/val2017.zip
  else
    echo "[coco] val2017.zip already present"
  fi

  if [[ ! -f "$root/annotations_trainval2017.zip" ]]; then
    echo "[coco] downloading annotations..."
    curl -fL -C - -o "$root/annotations_trainval2017.zip" \
      http://images.cocodataset.org/annotations/annotations_trainval2017.zip
  else
    echo "[coco] annotations zip already present"
  fi

  extract_zip_noclobber "$root/train2017.zip" "$root"
  extract_zip_noclobber "$root/val2017.zip" "$root"
  extract_zip_noclobber "$root/annotations_trainval2017.zip" "$root"
  "$PYTHON_FOR_PREP" scripts/prepare_coco2017.py --source-root "$root" --root "$root"
}

download_torchvision_weights() {
  export TORCH_HOME="${TORCH_HOME:-$ROOT/data/torch}"
  "$PYTHON_FOR_PREP" - <<'PY'
import torchvision.models as models

jobs = [
    ("resnet50", lambda: models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)),
    ("resnet152", lambda: models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2)),
    ("efficientnet_b0", lambda: models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)),
    ("vgg16", lambda: models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)),
]
for name, build in jobs:
    print(f"[torchvision] caching {name}...")
    build()
print("[torchvision] pretrained weights cached")
PY
}

link_shared_assets

if [[ "${DOWNLOAD_TORCHVISION_WEIGHTS:-0}" == "1" ]]; then
  download_torchvision_weights
fi

if [[ "${SKIP_EMBEDDINGS:-0}" != "1" ]]; then
  download_embeddings
fi

if [[ "${SKIP_FLICKR30K:-0}" != "1" ]]; then
  download_flickr30k
fi

if [[ "${SKIP_COCO:-0}" != "1" ]]; then
  download_coco2017
fi

echo "[done] data setup finished"
