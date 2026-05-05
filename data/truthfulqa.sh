#!/bin/bash
set -e

# 确保 Hugging Face 镜像生效：优先用项目 .env，否则默认国内镜像（uv run 子进程可能拿不到 .env）
# 脚本在 data/truthfulqa.sh，项目根为 odesteer/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [ -f "$PROJECT_ROOT/.env" ]; then
  set -a
  # shellcheck source=/dev/null
  source "$PROJECT_ROOT/.env"
  set +a
fi
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# 缓存到数据盘（若 .env 未设 HF_HOME，可按需改下方路径）
export HF_HOME="${HF_HOME:-/root/autodl-tmp/.cache/huggingface}"

# 离线时让模型也从本地缓存加载（不访问 Hugging Face）
if [ "${CRFDSteer_OFFLINE:-0}" = "1" ] || [ "${ODESteer_OFFLINE:-0}" = "1" ] || [ "${HF_HUB_OFFLINE:-0}" = "1" ]; then
  export HF_HUB_OFFLINE=1
fi

# format dataset（若存在 data/truthfulqa/truthful_qa_raw 会自动使用，无需设 TRUTHFULQA_LOCAL_DIR）
uv run python -u data/truthfulqa/format_dataset.py

# extract activations：若设置了 TRUTHFULQA_MODEL 则只跑该模型（可为本地路径或 Llama3.1-8B-Base 等）
if [ -n "${TRUTHFULQA_MODEL:-}" ]; then
  LAYER="${TRUTHFULQA_LAYER:-13}"
  uv run python -u data/truthfulqa/extract_activations.py -m "$TRUTHFULQA_MODEL" -l "$LAYER" -b 10
else
  uv run python -u data/truthfulqa/extract_activations.py -m Falcon-7B-Base -l 14 -b 10
  uv run python -u data/truthfulqa/extract_activations.py -m Mistral-7B-Base -l 15 -b 10
  uv run python -u data/truthfulqa/extract_activations.py -m Llama3.1-8B-Base -l 13 -b 10
  uv run python -u data/truthfulqa/extract_activations.py -m Qwen2.5-7B-Base -l 13 -b 10
fi