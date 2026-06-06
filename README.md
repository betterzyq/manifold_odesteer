# CRFDSteer (本仓库对 odesteer / ODESteer 的扩展与重命名)

本仓库在 [ODESteer](https://openreview.net/forum?id=CFewUmgIIL) 官方实现之上开展实验扩展（如 **ODE-Manifold**）；论文中的主 steer 方法命名为 **CRFDSteer**，对应 Hydra：`steer=CRFDSteer`，亦保留 `steer=ODESteer` 作为别名。

上游论文：**"ODESteer: A Unified ODE-Based Steering Framework for LLM Alignment"** (ICLR 2026)。核心思路仍基于常微分方程与障碍函数定义的向量场激活引导。

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
   （若使用本仓库维护者的 fork，将 URL 换为你的远程即可。）

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

7. **环境变量模板**

   项目根目录提供 `.env.example`，可复制为 `.env` 后按需填写（`GOOGLE_API_KEY`、`HF_ENDPOINT`、`HF_HOME` 等）。若使用 `uv run`，建议设置 `export UV_ENV_FILE=.env` 以便子进程自动加载（见步骤 4b）。

### 本地 Hub 解析与离线加载（实现说明摘要）

本仓库在 **`src/odesteer/lm/_hub_local.py`** 中实现了 Hugging Face **repo id → 本地 `hub` 下 snapshot 目录** 的解析，并在 **`HuggingFaceLM`**、**`utils/metric.py`**（TruthfulQA judge、质量模型、毒性评估前的环境加载）及 **`ultrafeedback_eval.py`** 等处统一使用 **`hf_pretrained_hub_kwargs`**：

- **`HF_HUB_CACHE`** 或 **`HF_HOME/hub`**：指定模型缓存的 `hub` 根目录（`models--org--repo` 所在层）。
- 若**未**设置上述变量，会依次尝试探测常见路径：`/root/autodl-tmp/.cache/huggingface/hub`、`~/.cache/huggingface/hub`（目录存在则使用），便于数据盘部署而无需手写 `.env`。
- 若已成功解析到带 `config.json` 的本地 snapshot，或设置 **`HF_HUB_OFFLINE=1`** / **`CRFDSteer_OFFLINE=1`**（或与旧脚本兼容的 **`ODESteer_OFFLINE=1`**） / **`CRFDSteer_LOCAL_ONLY=1`**（或 **`ODESteer_LOCAL_ONLY=1`**），则 **`local_files_only=True`**，避免再访问 Hub。
- **`HF_ENDPOINT`**（如 `https://hf-mirror.com`）须在 **`import transformers` 之前**写入环境变量；`_huggingface_lm.py` 与 **`ultrafeedback_eval.py`** 会在导入前加载项目根 `.env` 并设置默认镜像（未配置时）。

## ODE-Manifold（流形扩展）

在标准 **CRFDSteer** 上增加**切空间投影**与**保范数收回**的推理期扩展，详细步骤、CLI 与 Hydra 配置见 **[ODE_Manifold_使用教程.md](ODE_Manifold_使用教程.md)**。

## Quick Start

Here's a minimal example to run CRFDSteer on TruthfulQA with Llama-3.1-8B:

```bash
# 1. Prepare data
uv run bash data/truthfulqa.sh

# 2. Generate responses with CRFDSteer（亦可用 steer=ODESteer 别名）
uv run python -u scripts/truthfulqa/truthfulqa_generate.py \
    model=Llama3.1-8B-Base \
    layer_idx=13 \
    steer=CRFDSteer \
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
   Or: `export CRFDSteer_OFFLINE=1` (or legacy `ODESteer_OFFLINE=1`, same effect).

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

**Ultrafeedback 奖励模型（RM）**：评估依赖 **Skywork/Skywork-Reward-V2-Llama-3.1-8B**（与 `detox_eval` 所用的 Perspective / GPT2-XL 不同）。请将该模型完整缓存到上述 `hub` 目录，或设置 **`HF_HOME` / `HF_HUB_CACHE`**；离线时设置 **`HF_HUB_OFFLINE=1`**。也可用 **`-r /本地/snapshot路径`** 指定 RM 目录。若 RM 不在缓存且无法联网，评估会在加载阶段失败。

**重新生成与重新评估**：`ultrafeedback_generate.py` 在对应 **`raw_outputs/...jsonl` 已存在时会直接跳过**；`ultrafeedback_eval.py` 若统计 CSV 中已有同一 **Steering Method** 行也会跳过。需重跑时请删除（或改名）旧 **jsonl** 及 CSV 中对应行后再执行。

**Example - Compare CRFDSteer with baseline methods**:
```bash
# No steering (baseline)
uv run python -u scripts/ultrafeedback/ultrafeedback_generate.py \
    model=Llama3.1-8B-Base layer_idx=13 steer=NoSteer

# CAA
uv run python -u scripts/ultrafeedback/ultrafeedback_generate.py \
    model=Llama3.1-8B-Base layer_idx=13 steer=CAA steer.T=1.0

# CRFDSteer (our method)
uv run python -u scripts/ultrafeedback/ultrafeedback_generate.py \
    model=Llama3.1-8B-Base layer_idx=13 steer=CRFDSteer steer.T=5.0

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

### ODE-Manifold k 敏感度分析入口

已提供三个一键脚本（extract → generate → eval → plot）：

- `scripts/sensitivity/run_ode_manifold_k_sensitivity_truthfulqa.py`
- `scripts/sensitivity/run_ode_manifold_k_sensitivity_ultrafeedback.py`
- `scripts/sensitivity/run_ode_manifold_k_sensitivity_toxicity.py`

示例命令：

```bash
# TruthfulQA
uv run python scripts/sensitivity/run_ode_manifold_k_sensitivity_truthfulqa.py

# Ultrafeedback（可选指定 reward model）
uv run python scripts/sensitivity/run_ode_manifold_k_sensitivity_ultrafeedback.py \
    --reward-model Skywork/Skywork-Reward-V2-Llama-3.1-8B

# RealToxicityPrompts
uv run python scripts/sensitivity/run_ode_manifold_k_sensitivity_toxicity.py
```

仅重绘图（不重新抽取/生成/评估）：

```bash
uv run python scripts/sensitivity/run_ode_manifold_k_sensitivity_truthfulqa.py --plot-only
uv run python scripts/sensitivity/run_ode_manifold_k_sensitivity_ultrafeedback.py --plot-only
uv run python scripts/sensitivity/run_ode_manifold_k_sensitivity_toxicity.py --plot-only
```

### Multi-seed 完整网格（4 模型 × 3 数据集 × 8 方法，mean±std）

脚本 **`scripts/repro/run_multi_seed_grid.py`** 在多个 `--seeds` 下依次 **generate → eval**，并对各任务 `stat_results` 汇总 **mean±std**（输出至 `results/multi_seed_agg/`）。  
其中 **CRFDSteer / StepCRFDSteer** 按实验表启用 **ODE-Manifold**（`k`、`L`、`T` 按模型与任务写入脚本常量；Qwen 在 detox 上 **T=48**），并在缺少 `V_k` 时自动调用 **`data/extract_manifold.py`**（可用 `--skip-ensure-vk` 关闭）。  
可选 **`--crfd-t-json`** / **`--manifold-k-json`** 用 `Model` 或 `Model__dataset_key` 键覆盖默认值。

```bash
uv run python scripts/repro/run_multi_seed_grid.py --dry-run
uv run python scripts/repro/run_multi_seed_grid.py --aggregate-only
```


## Supported Models and Methods

### Models

| Model Name | HuggingFace Path | Recommended Layer |
|------------|-----------------|-------------------|
| `Llama3.1-8B-Base` | `meta-llama/Llama-3.1-8B` | 13 |
| `Mistral-7B-Base` | `mistralai/Mistral-7B-v0.3` | 15 |
| `Falcon-7B-Base` | `tiiuae/falcon-7b` | 14 |
| `Qwen2.5-7B-Base` | `Qwen/Qwen2.5-7B` | 13 |

**Note:** The steering layer indices reported in our paper differ slightly from those used here, as this implementation starts counting from zero.

### CRFDSteer: range of \(T\) (paper Table 5)

复现实验时，**CRFDSteer** 的 **`steer.T`** 建议在 **ODESteer 原论文 Table 5** 给出的区间内选取（或在该区间内网格搜索）。与上表模型名的对应关系如下：

| HuggingFace checkpoint | 代码中 `model=` | Range of \(T\) |
|------------------------|-----------------|----------------|
| `meta-llama/Llama-3.1-8B` | `Llama3.1-8B-Base` | 4–6 |
| `mistralai/Mistral-7B-v0.3` | `Mistral-7B-Base` | 3–4 |
| `tiiuae/falcon-7b` | `Falcon-7B-Base` | 20–23 |
| `Qwen/Qwen2.5-7B` | `Qwen2.5-7B-Base` | 13–16（**去毒 / RealToxicityPrompts 任务建议约 48**） |

其他 steering 方法：`RepE` / `CAA` / `ITI` 中 **`T` 为向量加法系数** `X + T·v`，默认常从 **`steer.T=1.0`** 开始；**`MiMiC` / `LinAcT`** 当前实现**未使用** `T`；**`NoSteer`** 无需设置 **`steer.T`**。详见各 `confs/steer/*.yaml`。

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

**CRFDSteer (our methods)**:
- `CRFDSteer` - Barrier function-guided ODE Steering (main method; legacy name `ODESteer` still works)
- `StepCRFDSteer` - One-step variant (legacy `StepODESteer`)

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
├── .env.example            # 环境变量模板（复制为 .env，勿提交密钥）
├── results/                # Generated outputs (auto-created, gitignored)
└── pyproject.toml          # Project dependencies
```

**Git 与大数据**：`.gitignore` 默认忽略 `results/`、各数据集的 `texts/`、`activations/`、体积较大的 `data/toxicity/jigsaw/` 与 `data/toxicity/real_tox_prompts/` 等。大文件**不会**随 `git checkout` 恢复，需自行备份或使用 DVC / Git LFS 等方案。

## Citation

若使用本仓库的 CRFDSteer / ODE-Manifold 扩展，请在正文中写明方法名；引用上游基础方法时请使用以下 **ODESteer（ICLR 2026）** BibTeX。

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
