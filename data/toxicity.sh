#!/usr/bin/env bash
# 毒性数据准备：Jigsaw（Kaggle）+ Real Toxicity Prompts（Hugging Face）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# 与 truthfulqa.sh 一致：加载 .env（Kaggle 凭据在 ~/.kaggle/kaggle.json，此处可配 HF 镜像等）
if [ -f "$PROJECT_ROOT/.env" ]; then
  set -a
  # shellcheck source=/dev/null
  source "$PROJECT_ROOT/.env"
  set +a
fi
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/.cache/huggingface}"

if [ "${CRFDSteer_OFFLINE:-0}" = "1" ] || [ "${ODESteer_OFFLINE:-0}" = "1" ] || [ "${HF_HUB_OFFLINE:-0}" = "1" ]; then
  export HF_HUB_OFFLINE=1
fi

# 使用 uv run kaggle，确保走当前项目的 .venv。若报错 bad interpreter（venv 曾从别的路径复制），在项目根执行：
#   uv pip install --force-reinstall kaggle
# Kaggle 凭据：~/.kaggle/kaggle.json
KAGGLE_DL=(uv run kaggle competitions download -c jigsaw-unintended-bias-in-toxicity-classification)
ZIP_NAME="jigsaw-unintended-bias-in-toxicity-classification.zip"
JIGSAW_STAGING="data/toxicity/jigsaw-unintended-bias-in-toxicity-classification"
JIGSAW_DIR="data/toxicity/jigsaw"

if [ ! -f "$JIGSAW_DIR/all_data.csv" ]; then
  if [ ! -f "$ZIP_NAME" ]; then
    echo "→ Downloading Jigsaw dataset from Kaggle (requires ~/.kaggle/kaggle.json) ..."
    "${KAGGLE_DL[@]}"
  else
    echo "→ Found existing $ZIP_NAME, skipping Kaggle download."
  fi
  if [ ! -f "$ZIP_NAME" ]; then
    echo "Error: $ZIP_NAME not found after download. Check Kaggle API credentials and network." >&2
    exit 1
  fi
  unzip -o "$ZIP_NAME" -d "$JIGSAW_STAGING"
  rm -f "$ZIP_NAME"
  mkdir -p "$JIGSAW_DIR"
  mv "$JIGSAW_STAGING"/* "$JIGSAW_DIR"/
  rmdir "$JIGSAW_STAGING" 2>/dev/null || true
  echo "✓ Jigsaw files unpacked to $JIGSAW_DIR"
else
  echo "✓ $JIGSAW_DIR/all_data.csv already present, skipping Jigsaw download/unpack."
fi

uv run python -u data/toxicity/preprocess_jigsaw.py
uv run python -u data/toxicity/preprocess_rtp.py
uv run python -u data/toxicity/format_dataset.py

uv run python -u data/toxicity/extract_activations.py -m Falcon-7B-Base -l 14 -b 10
uv run python -u data/toxicity/extract_activations.py -m Mistral-7B-Base -l 15 -b 10
uv run python -u data/toxicity/extract_activations.py -m Llama3.1-8B-Base -l 13 -b 10
uv run python -u data/toxicity/extract_activations.py -m Qwen2.5-7B-Base -l 13 -b 10
