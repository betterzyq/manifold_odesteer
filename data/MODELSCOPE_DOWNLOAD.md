# ModelScope 下载到数据盘（HF Hub 缓存布局）

数据盘：`/root/autodl-tmp`  
Hub 根目录：`HF_HOME=/root/autodl-tmp/.cache/huggingface`，实际模型在 **`$HF_HOME/hub/`** 下。

**务必使用绝对路径**，不要用 `./root/autodl-tmp/...`（会变成系统盘 `/root/root/...`）。

---

## 通用步骤（每个模型）

```bash
export HF_HOME=/root/autodl-tmp/.cache/huggingface
REPO="组织/模型名"   # 如 allenai/truthfulqa-truth-judge-llama2-7B
DIR_NAME="models--${REPO//\//--}"
BASE="$HF_HOME/hub/$DIR_NAME"
mkdir -p "$BASE/snapshots/main"
mkdir -p "$BASE/refs"
echo main > "$BASE/refs/main"
modelscope download --model "$REPO" --local_dir "$BASE/snapshots/main"
```

下载完成后，代码会通过 `hf_pretrained_hub_kwargs` 解析到 `snapshots/main`，并在检测到本地 `config.json` 时设置 **`local_files_only=True`**，**不再访问 Hugging Face**。

可选全局强制仅本地（无网环境）：

```bash
export CRFDSteer_LOCAL_ONLY=1
# 或与旧脚本兼容：
# export ODESteer_LOCAL_ONLY=1
```

---

## TruthfulQA 评测用 Judge（两个）

与代码中默认 repo id 一致（也可用环境变量覆盖，见下）。

### Truth Judge

```bash
export HF_HOME=/root/autodl-tmp/.cache/huggingface
BASE="$HF_HOME/hub/models--allenai--truthfulqa-truth-judge-llama2-7B"
mkdir -p "$BASE/snapshots/main" "$BASE/refs"
echo main > "$BASE/refs/main"
modelscope download --model allenai/truthfulqa-truth-judge-llama2-7B --local_dir "$BASE/snapshots/main"
```

### Info Judge

```bash
export HF_HOME=/root/autodl-tmp/.cache/huggingface
BASE="$HF_HOME/hub/models--allenai--truthfulqa-info-judge-llama2-7B"
mkdir -p "$BASE/snapshots/main" "$BASE/refs"
echo main > "$BASE/refs/main"
modelscope download --model allenai/truthfulqa-info-judge-llama2-7B --local_dir "$BASE/snapshots/main"
```

---

## Quality 指标用 GPT2-XL（与 ModelScope 一致）

代码默认使用 **`openai-community/gpt2-xl`**（对应目录 `models--openai-community--gpt2-xl`），与下列命令一致：
    
```bash
export HF_HOME=/root/autodl-tmp/.cache/huggingface
BASE="$HF_HOME/hub/models--openai-community--gpt2-xl"
mkdir -p "$BASE/snapshots/main" "$BASE/refs"
echo main > "$BASE/refs/main"
modelscope download --model openai-community/gpt2-xl --local_dir "$BASE/snapshots/main"
```

若需改用其它路径或 repo，可设置：

```bash
export CRFDSteer_QUALITY_MODEL="openai-community/gpt2-xl"
```

---

## 环境变量一览（评测相关）

| 变量 | 含义 |
|------|------|
| `CRFDSteer_TRUTHFULQA_TRUTH_JUDGE` | Truth judge 的 HF repo id（默认 `allenai/truthfulqa-truth-judge-llama2-7B`）；亦可使用旧名 `ODESteer_TRUTHFULQA_TRUTH_JUDGE` |
| `CRFDSteer_TRUTHFULQA_INFO_JUDGE` | Info judge 的 HF repo id（默认 `allenai/truthfulqa-info-judge-llama2-7B`）；亦可使用旧名 `ODESteer_TRUTHFULQA_INFO_JUDGE` |
| `CRFDSteer_QUALITY_MODEL` | PPL/Dist 模型（默认 `openai-community/gpt2-xl`）；亦可使用旧名 `ODESteer_QUALITY_MODEL` |
| `CRFDSteer_LOCAL_ONLY=1` | 所有 `from_pretrained` 仅本地，不访问 Hub（旧名 `ODESteer_LOCAL_ONLY` 仍有效） |
| `CRFDSteer_DEBUG_HUB=1` | 打印解析到的本地 snapshot 路径（旧名 `ODESteer_DEBUG_HUB` 仍有效） |
| `CRFDSteer_OFFLINE=1` | 强制离线加载（旧名 `ODESteer_OFFLINE` 仍有效） |

---

## Mistral 等主模型（参考）

```bash
modelscope download --model mistralai/Mistral-7B-v0.3 --local_dir /root/autodl-tmp/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.3/snapshots/main
```

（同样先 `mkdir` + `refs/main` 写入 `main`。）
