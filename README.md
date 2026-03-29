# ODESteer: A Unified ODE-Based Steering Framework for LLM Alignment

Official implementation of the paper **"ODESteer: A Unified ODE-Based Steering Framework for LLM Alignment"** (ICLR 2026)

This repository provides code for ODESteer, a novel activation steering approach based on ordinary differential equations and barrier functions from control theory.

[Paper](https://openreview.net/forum?id=CFewUmgIIL), [Website](https://odesteer.github.io/)

![Thumbnail](assets/thumbnail.png)

## Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-compatible GPU (recommended for running LLMs)
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

1. **Install uv** (if not already installed): please follow instructions in https://docs.astral.sh/uv/getting-started/installation/

2. **Clone the repository**:
   ```bash
   git clone git@github.com:ZhaoHongjue/odesteer.git
   cd odesteer
   ```

3. **Install dependencies**:
   ```bash
   uv sync
   ```
   This will automatically create a virtual environment and install all required packages.

4. **Set up API keys** (required for detoxification evaluation):

   For toxicity evaluation, you need a [Perspective API](https://perspectiveapi.com/) key:

   a. Create a `.env` file in the project root:
   ```bash
   echo "GOOGLE_API_KEY=your_api_key_here" > .env
   ```

   b. Export the environment file so that `uv run` loads it:
   ```bash
   export UV_ENV_FILE=.env
   ```
   To make this permanent, add the above line to your `~/.bashrc` or `~/.zshrc`.

5. **Hugging Face 国内镜像（中国大陆用户）**

   若无法访问 huggingface.co，请在项目根目录的 `.env` 中加入（或使用已配置好的 `.env`）：
   ```
   HF_ENDPOINT=https://hf-mirror.com
   ```
   并确保已设置 `UV_ENV_FILE` 指向该 `.env`（见步骤 4b），这样 `uv run` 下所有模型与数据集下载会走 [hf-mirror](https://hf-mirror.com)，长期生效。

6. **模型/数据集缓存到数据盘（系统盘空间不足时）**

   默认缓存目录为 `~/.cache/huggingface`（在系统盘）。若租赁服务器系统盘小、数据盘大，可在 `.env` 中设置：
   ```
   HF_HOME=/root/autodl-tmp/.cache/huggingface
   ```
   将路径改为你的数据盘挂载点（如 `/data`、`/mnt/data` 等），模型与数据集会下载到该目录下，避免占满系统盘。

## Quick Start

Here's a minimal example to run ODESteer on TruthfulQA with Llama-3.1-8B:

```bash
# 1. Prepare data
uv run bash data/truthfulqa.sh

# 2. Generate responses with ODESteer
uv run python -u scripts/truthfulqa/truthfulqa_generate.py \
    model=Llama3.1-8B-Base \
    layer_idx=13 \
    steer=ODESteer \
    steer.T=5.0

# 3. Evaluate results
uv run python -u scripts/truthfulqa/truthfulqa_eval.py -m Llama3.1-8B-Base -l 13 -d
```

## Data Preparation

```bash
uv run bash data/ultrafeedback.sh
uv run bash data/truthfulqa.sh
uv run bash data/toxicity.sh
```

These scripts will:
- Download and format the each dataset
- Extract activations from the specified model at the specified layer

**Note**: For toxicity data preparation, you need Kaggle API credentials configured (`~/.kaggle/kaggle.json`).

### Offline / No Network (无网络或离线)

When Hugging Face is unreachable (e.g. `Network is unreachable`):

1. **Use local cache only** (dataset and models must be cached first, e.g. from a previous run with network):
   ```bash
   export HF_HUB_OFFLINE=1
   uv run bash data/truthfulqa.sh
   ```
   Or: `export ODESteer_OFFLINE=1` (same effect).

2. **Use a local model path** instead of downloading: pass the path to the model directory with `-m` when running scripts (e.g. `-m /path/to/Llama-3.1-8B`). The dataset still needs to be in cache or loaded once with network.

3. **TruthfulQA 数据**：若已用 ModelScope 下载到 `data/truthfulqa/truthful_qa_raw`，无需设置 `TRUTHFULQA_LOCAL_DIR`，格式化脚本会自动使用该目录。只跑一个本地模型时：`TRUTHFULQA_MODEL=/path/to/Llama-3.1-8B TRUTHFULQA_LAYER=13 uv run bash data/truthfulqa.sh`（并设 `HF_HUB_OFFLINE=1` 若需强制离线）。

### Where to Put Existing Data and Models (已有数据/模型放置位置)

Assume project root is `odesteer/` (the repo directory).

| 类型 | 放置位置 | 说明 |
|------|----------|------|
| **TruthfulQA 已格式化文本** | `odesteer/data/truthfulqa/texts/` | 放入 `pos_0.jsonl`, `pos_1.jsonl`, `neg_0.jsonl`, `neg_1.jsonl`（每行 JSON 含 `idx`, `question`, `answer`）。放好后可**跳过** `format_dataset.py`，直接运行 `extract_activations.py`。 |
| **TruthfulQA（ModelScope 下载）** | 建议 `odesteer/data/truthfulqa/truthful_qa_raw` | 使用 `modelscope download --dataset evalscope/truthful_qa --local_dir ./data/truthfulqa/truthful_qa_raw`（在项目根 `odesteer/` 下执行）。下载后设置 `export TRUTHFULQA_LOCAL_DIR=./data/truthfulqa/truthful_qa_raw`，再运行 `uv run python -u data/truthfulqa/format_dataset.py`，会从该目录读取并生成 `texts/` 下的 jsonl。 |
| **TruthfulQA 原始数据集缓存** | `~/.cache/huggingface/datasets/` | 从有网络的机器复制整个 `datasets/` 目录到此路径，保持内部结构（如 `truthfulqa___truthful_qa/`）。然后设置 `HF_HUB_OFFLINE=1` 再跑 `format_dataset.py`。 |
| **模型（任意路径）** | 任意目录，如 `/root/models/Llama-3.1-8B` | 目录内需包含 `config.json`、权重文件等。运行脚本时用 `-m /root/models/Llama-3.1-8B` 指定。提取的 activations 会写入 `odesteer/data/truthfulqa/activations/<模型名或路径名>/`。 |
| **模型（HF 缓存格式）** | `~/.cache/huggingface/hub/` | 从别处复制 `hub/` 下对应模型目录（如 `models--meta-llama--Llama-3.1-8B/`），设置 `HF_HUB_OFFLINE=1`，脚本中仍用 `-m Llama3.1-8B-Base`。 |
| **提取好的 activations** | `odesteer/data/truthfulqa/activations/<ModelName>/` | 例如 `pos_0_activations_layer13.pt`, `pos_0_question_idx.pt`, `neg_0_activations_layer13.pt` 等。与 `texts/` 中的文件名对应，后续评估/生成会从这里读。 |

## Running Experiments

### Helpfulness (Ultrafeedback)

**Generate responses**:
```bash
uv run python -u scripts/ultrafeedback/ultrafeedback_generate.py \
    model=<MODEL> \
    layer_idx=<LAYER> \
    steer=<STEER> \
    steer.T=<T_VALUE>
```

**Evaluate responses**:
```bash
uv run python -u scripts/ultrafeedback/ultrafeedback_eval.py -m <MODEL> -l <LAYER> -d
```
The command will automatically evaluate all steering methods with the specific model and layer.

**Example - Compare ODESteer with baseline methods**:
```bash
# No steering (baseline)
uv run python -u scripts/ultrafeedback/ultrafeedback_generate.py \
    model=Llama3.1-8B-Base layer_idx=13 steer=NoSteer

# CAA
uv run python -u scripts/ultrafeedback/ultrafeedback_generate.py \
    model=Llama3.1-8B-Base layer_idx=13 steer=CAA steer.T=1.0

# ODESteer (our method)
uv run python -u scripts/ultrafeedback/ultrafeedback_generate.py \
    model=Llama3.1-8B-Base layer_idx=13 steer=ODESteer steer.T=5.0

# Evaluate
uv run python -u scripts/ultrafeedback/ultrafeedback_eval.py -m Llama3.1-8B-Base -l 13 -d
```

### Truthfulness (TruthfulQA)

**Generate responses**:
```bash
uv run python -u scripts/truthfulqa/truthfulqa_generate.py \
    model=<MODEL> \
    layer_idx=<LAYER> \
    steer=<STEER> \
    steer.T=<T_VALUE>
```

**Evaluate responses**:
```bash
uv run python -u scripts/truthfulqa/truthfulqa_eval.py -m <MODEL> -l <LAYER> -d
```


### Detoxification (RealToxicityPrompts)

**Generate responses**:
```bash
uv run python -u scripts/toxicity/detox_generate.py \
    model=<MODEL> \
    layer_idx=<LAYER> \
    steer=<STEER> \
    steer.T=<T_VALUE>
```

**Evaluate responses**:
```bash
uv run python -u scripts/toxicity/detox_eval.py -m <MODEL> -l <LAYER> -d
```


## Supported Models and Methods

### Models

| Model Name | HuggingFace Path | Recommended Layer |
|------------|-----------------|-------------------|
| `Llama3.1-8B-Base` | `meta-llama/Llama-3.1-8B` | 13 |
| `Mistral-7B-Base` | `mistralai/Mistral-7B-v0.1` | 15 |
| `Falcon-7B-Base` | `tiiuae/falcon-7b` | 14 |
| `Qwen2.5-7B-Base` | `Qwen/Qwen2.5-7B` | 13 |

**Note:** The steering layer indices reported in our paper differ slightly from those used here, as this implementation starts counting from zero.

### Steering Methods

**Baselines:**

- `NoSteer` - No steering (baseline)
- `RepE` - [Paper](http://arxiv.org/abs/2310.01405), [Code](https://github.com/andyzoujm/representation-engineering)
- `CAA` - [Paper](https://aclanthology.org/2024.acl-long.828/), [Code](https://github.com/nrimsky/CAA)
- `ITI` - [Paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/81b8390039b7302c909cb769f8b6cd93-Abstract-Conference.html), [Code](https://github.com/likenneth/honest_llama)
- `MiMiC` - [Paper](https://openreview.net/forum?id=GwA4go0Mw4), [Code](https://github.com/shauli-ravfogel/affine-steering)
- `LinAcT` - [Paper](https://openreview.net/forum?id=l2zFn6TIQi), [Code](https://github.com/apple/ml-act)
- `HPR` - [Paper](https://aclanthology.org/2024.emnlp-main.761/), [Code](https://github.com/VinAIResearch/HPR)
- `RE-Control` - [Paper](http://arxiv.org/abs/2406.05954), [Code](https://github.com/Lingkai-Kong/RE-Control)
- `TruthFlow`- [Paper](http://arxiv.org/abs/2502.04556), [Code](https://github.com/wwwhy725/TruthFlow)

**ODESteer (our methods)**:
- `ODESteer` - Barrier function-guided ODE Steering (main method)
- `StepODESteer` - One-step ODESteer variant

**Note:** For DNN-based steering methods (HPR, RE-Control, and TruthFlow), please refer to corresponding official repos for detailed implementation.


## Project Structure

```
odesteer/
├── src/odesteer/           # Main package
│   ├── lm/                 # Language model wrappers
│   ├── steer/              # Steering methods implementation
│   └── utils/              # Utilities (kernels, metrics, data loading)
├── scripts/                # Experiment scripts
│   ├── ultrafeedback/      # Ultrafeedback generation & evaluation
│   ├── truthfulqa/         # TruthfulQA generation & evaluation
│   └── toxicity/           # Detoxification generation & evaluation
├── data/                   # Data preparation scripts
│   ├── ultrafeedback/      # Ultrafeedback data preprocessing
│   ├── truthfulqa/         # TruthfulQA data processing
│   └── toxicity/           # Toxicity data processing
├── confs/                  # Hydra configuration files
├── results/                # Generated outputs (auto-created)
└── pyproject.toml          # Project dependencies
```

## Citation

```bibtex
@inproceedings{zhao2026odesteer,
      title={ODESteer: A Unified ODE-Based Steering Framework for LLM Alignment},
      author={Hongjue Zhao* and Haosen Sun* and Jiangtao Kong and
               Xiaochang Li and Qineng Wang and Liwei Jiang and Qi Zhu and 
               Tarek F. Abdelzaher and Yejin Choi and Manling Li† and Huajie Shao†},
      booktitle={The Fourteenth International Conference on Learning Representations},
      year={2026},
      url={https://odesteer.github.io/},
}
```
