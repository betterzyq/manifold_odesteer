# ODE-Manifold 使用教程

本教程说明如何在 CRFDSteer 代码库中使用 **ODE-Manifold**（基于流形边界势垒的常微分方程引导）：在固定低秩子空间上对向量场做**切空间投影**，并在每步（或积分结束后）用**保范数收回（Retraction）**稳定隐状态范数，减轻大幅干预导致的困惑度飙升或生成异常。  
当前版本中，这套逻辑已可通用于多数 steer（不仅是 CRFDSteer / StepCRFDSteer）。

> 前置条件与安装、HF 镜像与缓存、离线加载等通用说明见仓库根目录 **[README.md](README.md)**。本教程默认你已完成 `uv sync`，且已按 README 准备好模型与数据。

---

## 1. 方法在代码里做什么

ODE-Manifold 目前通过基类后处理接入，已覆盖：

- `CRFDSteer` / `StepCRFDSteer`
- `CAA` / `RepE` / `ITI`
- `LinAcT` / `MiMiC`

实现入口见 `src/odesteer/steer/_base_steer.py` 与 `src/odesteer/steer/_manifold.py`（ODE 两类仍保留原有路径）。

1. **切空间投影**  
   先按原 CRFDSteer 计算归一化梯度方向 `vf`（与论文一致），若提供了正交基矩阵 `V_k ∈ ℝ^{d×k}`，则替换为：
   \[
   v_f \leftarrow V_k V_k^\top v_f
   \]
   即将向量场限制在 `V_k` 张成的子空间内。

2. **保范数收回**  
   多步积分结束（`CRFDSteer`）或单步更新后（`StepCRFDSteer`），将新激活 `X_new` 按与更新前 `X` 相同的 L2 范数缩放：
   \[
   X_{\mathrm{new}} \leftarrow X_{\mathrm{new}} \cdot \frac{\lVert X \rVert_2}{\lVert X_{\mathrm{new}} \rVert_2 + \varepsilon}
   \]

**与本仓库 CRFDSteer 核心的关系**：分类器（多项式计数草图 + 逻辑回归）、障碍函数与 `torchdiffeq.odeint` 的调用方式不变；仅在向量场与输出激活上增加上述两步。对其他 steer（CAA / RepE / ITI / LinAcT / MiMiC）则是通过统一后处理复用同样的投影与收回逻辑。

---

## 2. 整体流程（建议顺序）

```
准备任务数据与激活 (.pt)
        ↓
离线 SVD 提取 V_k（extract_manifold.py）
        ↓
生成脚本 + 任一支持的 steer + 流形参数
        ↓
（可选）按原 README 运行对应 eval 脚本
```

**重要约定**：

- **`layer_idx`（引导层）** 必须与提取激活、提取 `V_k` 时使用的 **层号一致**。
- **`model` 短名**（如 `Llama3.1-8B-Base`）必须与 `data/.../activations/<model>/` 目录名一致，否则 `extract_manifold.py` 找不到文件。

---

## 3. 阶段 A：准备激活（与原版相同）

ODE-Manifold 的 `V_k` 来自**已有层激活**，需先完成对应任务的 `extract_activations`（或等价流程），使目录下出现：

`data/<task>/activations/<model>/*_activations_layer<L>.pt`

其中 `<task>` 可为 `truthfulqa`、`ultrafeedback`、`toxicity` 之一（与主 README 中数据准备一致）。

示例（TruthfulQA）：

```bash
cd /path/to/odesteer
uv run bash data/truthfulqa.sh
```

若仅本地 Hub、无网络，请按 README「Offline / No Network」设置 `HF_HUB_OFFLINE=1` 等，并保证模型与数据已在缓存或磁盘路径中。

---

## 4. 阶段 B：离线提取流形基 `V_k`

使用脚本 **`data/extract_manifold.py`**：在指定 `activations` 目录下匹配所有 `*activations_layer{L}.pt`，沿样本维拼接为矩阵 `X ∈ ℝ^{N×d}`，对 `X` 做 `torch.linalg.svd`，取右奇异向量前 `k` 个，保存为：

\[
V_k = V_h[:k]^\top \in \mathbb{R}^{d \times k}
\]

默认输出路径：

`data/<task>/activations/<model>/manifold_Vk_layer<L>_k<K>.pt`

### 4.1 命令行参数说明

| 参数 | 说明 |
|------|------|
| `-m` / `--model` | 与激活子目录名一致，默认 `Llama3.1-8B-Base` |
| `-l` / `--layer_idx` | **必填**，目标层索引（与 steering 层一致） |
| `-k` / `--manifold_dim` | 保留的奇异向量个数 `k`，默认 `128` |
| `--task` | `truthfulqa` \| `ultrafeedback` \| `toxicity`，决定默认 `data/<task>/activations/...` |
| `-o` / `--output` | 自定义输出 `.pt` 路径 |
| `--activations_dir` | 直接指定含 `*_activations_layer{L}.pt` 的目录（覆盖 `--task` 默认路径） |

### 4.2 示例命令

**TruthfulQA，层 13，k=128：**

```bash
uv run python data/extract_manifold.py \
  -m Llama3.1-8B-Base \
  -l 13 \
  -k 128 \
  --task truthfulqa
```

**UltraFeedback：**

```bash
uv run python data/extract_manifold.py \
  -m Llama3.1-8B-Base \
  -l 13 \
  -k 128 \
  --task ultrafeedback
```

**毒性任务（Jigsaw 激活目录）：**

```bash
uv run python data/extract_manifold.py \
  -m Llama3.1-8B-Base \
  -l 13 \
  -k 128 \
  --task toxicity
```

**自定义激活目录（例如你从别处拷贝的 `activations`）：**

```bash
uv run python data/extract_manifold.py \
  -l 13 \
  -k 128 \
  --activations_dir /path/to/activations/Llama3.1-8B-Base
```

### 4.3 输出文件格式

- 默认可用 `torch.load` 读入为 **`FloatTensor`，形状 `[d, k]`**，默认文件名为 `manifold_Vk_layer<L>_k<K>.pt`，不同 `k` 不会互相覆盖。
- 若你保存的是字典且含键 **`V_k`**，生成端合并 CLI 时也会自动解包（见 `manifold_cli.py`）。

若请求的 `k` 大于 `min(N, d)`，脚本会自动将有效列数降为 `k_eff` 并在终端打印警告。

---

## 5. 阶段 C：生成时启用 ODE-Manifold（通用于多 steer）

支持两种方式：**Hydra 配置**（推荐与实验记录一致）与 **额外命令行标志**（与 Hydra 混用）。  
当前支持的 steer（`CRFDSteer`、`StepCRFDSteer`、`CAA`、`RepE`、`ITI`、`LinAcT`、`MiMiC`）都可通过 `steer_model_kwargs` 传入相同流形字段。

**重要（Hydra 命令行）**：这些字段在 YAML 里都写在 **`steer.kwargs`** 下面，与 `solver`、`n_components` 同级。因此命令行覆盖时必须写 **`steer.kwargs.use_manifold=true`**，**不能**写 `steer.use_manifold=true`（否则会报 `Key 'use_manifold' is not in struct`）。`steer.T`、`steer.name` 仍在 `steer` 顶层。

### 5.1 配置项含义

下列键在配置文件中的路径为 **`steer.kwargs.<键名>`**（Hydra 覆盖写法见下节示例）。

| 字段 | 类型 | 说明 |
|------|------|------|
| `use_manifold` | bool | 必须为 `true` 才会在仅提供 `vk_path` 时从磁盘加载 `V_k`；为 `true` 且未提供有效基时会报错 |
| `vk_path` | str / null | `V_k` 的 `.pt` 文件路径（与 `use_manifold=true` 联用） |
| `V_k` | Tensor（可选） | 若由外部代码注入，可直接传张量；通常由 CLI 合并逻辑从 `vk_path` 加载 |
| `manifold_dim` | int | 加载后对 `V_k` 按列截断的上限；若文件列数小于该值则不变 |
| `use_manifold_retraction` | bool | **保范数收回**：积分/单步后按输入范数缩放。与 `use_manifold`/`V_k` **无关**时，表示 **欧氏 CRFDSteer + 仅收回**（消融：探究投影的作用）；与 `use_manifold` 联用时见下表 |

**注意**：仅配置 `vk_path` 而不将 `use_manifold` 设为 `true` 时，**不会**从路径加载 `V_k`，避免误开流形。

**消融对应关系（CLI / Hydra，见 5.3）**（以下以 CRFDSteer 记号说明，其他 steer 同理）：

1. **Vanilla CRFDSteer / StepCRFDSteer**：不加流形相关参数。  
2. **StepCRFDSteer + 投影 + 收回** 或 **CRFDSteer + 投影 + 收回**：`use_manifold` + `vk_path` + `use_manifold_retraction=true`。  
3. **CRFDSteer + 仅投影**：`use_manifold` + `vk_path`，`use_manifold_retraction=false`。  
4. **CRFDSteer + 仅保范数（无投影，欧氏梯度 + 收回）**：只设 **`steer.kwargs.use_manifold_retraction=true`**，**不要**设 `use_manifold` / `vk_path`（或 CLI 仅 `--use_manifold_retraction`）。输出文件名自动带 **`-NormRetr`**，与 Vanilla 区分。

已接入流形 CLI 合并的入口脚本：

- `scripts/truthfulqa/truthfulqa_generate.py`
- `scripts/ultrafeedback/ultrafeedback_generate.py`
- `scripts/toxicity/detox_generate.py`

评估脚本仍只读取已生成的 `jsonl`，**无需**再传流形参数。

### 5.2 方式一：纯 Hydra（推荐）

TruthfulQA 示例（路径请换成你机器上的实际文件）。**完整 Manifold（投影 + 收回）**请显式打开 `use_manifold_retraction`：

```bash
uv run python -u scripts/truthfulqa/truthfulqa_generate.py \
  model=Llama3.1-8B-Base \
  layer_idx=13 \
  steer=CRFDSteer \
  steer.T=5.0 \
  steer.kwargs.use_manifold=true \
  steer.kwargs.use_manifold_retraction=true \
  steer.kwargs.vk_path=/绝对路径/odesteer/data/truthfulqa/activations/Llama3.1-8B-Base/manifold_Vk_layer13_k128.pt \
  steer.kwargs.manifold_dim=128
```

**仅投影（消融）**：省略 `steer.kwargs.use_manifold_retraction` 或设为 `false`（与 CLI 仅 `--use_manifold` 行为一致）。

UltraFeedback / 毒性任务：将脚本与顶层 `config_name` 换成 `ultrafeedback`、`detox`（与 README 一致），流形相关覆盖仍使用 **`steer.kwargs.*`**。

**StepCRFDSteer**：

```bash
uv run python -u scripts/truthfulqa/truthfulqa_generate.py \
  model=Llama3.1-8B-Base \
  layer_idx=13 \
  steer=StepCRFDSteer \
  steer.T=1.0 \
  steer.kwargs.use_manifold=true \
  steer.kwargs.vk_path=/path/to/manifold_Vk_layer13_k128.pt
```

**CAA（示例：同样可启用投影+收回）**：

```bash
uv run python -u scripts/truthfulqa/truthfulqa_generate.py \
  model=Llama3.1-8B-Base \
  layer_idx=13 \
  steer=CAA \
  steer.T=1.0 \
  steer.kwargs.use_manifold=true \
  steer.kwargs.use_manifold_retraction=true \
  steer.kwargs.vk_path=/path/to/manifold_Vk_layer13_k128.pt \
  steer.kwargs.manifold_dim=128
```

### 5.3 方式二：流形相关 CLI（与 Hydra 混用）

以下参数在脚本导入阶段会从 `sys.argv` 中**剥离**后再交给 Hydra，因此可与 `model=...` 等任意混排（多行续写时每行末尾需有 `\`，最后一行除外）。

| 标志 | 说明 |
|------|------|
| `--use_manifold` | 开启流形：加载 `V_k`、启用**切空间投影**；**默认不**开启保范数收回（仅投影消融） |
| `--use_manifold_retraction` | 与 `--use_manifold` 联用：开启**保范数收回**（完整 ODE-Manifold，生成通常更稳） |
| `--vk_path` | `V_k` 文件路径；若省略则尝试使用 Hydra 里已配置的 **`steer.kwargs.vk_path`** |
| `--manifold_dim` | 默认 `128`；对加载的 `V_k` 按列截断 |

**仅投影（消融）**：

```bash
uv run python -u scripts/truthfulqa/truthfulqa_generate.py \
  --use_manifold \
  --vk_path /path/to/manifold_Vk_layer13_k128.pt \
  --manifold_dim 128 \
  model=Llama3.1-8B-Base \
  layer_idx=13 \
  steer=CRFDSteer \
  steer.T=5.0
```

**投影 + 收回（推荐）**：

```bash
uv run python -u scripts/truthfulqa/truthfulqa_generate.py \
  --use_manifold \
  --use_manifold_retraction \
  --vk_path /path/to/manifold_Vk_layer13_k128.pt \
  --manifold_dim 128 \
  model=Llama3.1-8B-Base \
  layer_idx=13 \
  steer=CRFDSteer \
  steer.T=5.0
```

**欧氏 CRFDSteer + 仅保范数收回（消融 4，无 `V_k`）** — 输出带 **`-NormRetr`**：

```bash
uv run python -u scripts/truthfulqa/truthfulqa_generate.py \
  model=Llama3.1-8B-Base \
  layer_idx=13 \
  steer=CRFDSteer \
  steer.T=5.0 \
  steer.kwargs.use_manifold_retraction=true
```

或 CLI：`--use_manifold_retraction`（不要加 `--use_manifold`）。

若 `--use_manifold` 已加，但未提供 `--vk_path` 且配置里也没有 **`steer.kwargs.vk_path`**，运行时会抛出明确错误。

---

## 6. 与主 README 的交叉引用

- **论文中 CRFDSteer 的 `T` 取值**：仍请参考主 README「CRFDSteer: range of T (paper Table 5)」；ODE-Manifold 不改变 `T` 的含义，但投影与收回会改变实际轨迹，可能需要小幅重调 `T`。
- **输出文件名与原版共存**：流形实验为 **`-Manifold-k{k}`** / **`-Manifold-k{k}-proj`**；**仅欧氏 + 收回** 为 **`-NormRetr`**，与 Vanilla 及彼此区分，避免跳过或覆盖评估行。
- **自定义命名**：若已在 `steer.name` 里手写包含 `Manifold-k{k}`，脚本**不会**重复追加（见 `steer_name_with_manifold_suffix`）。
- **跳过已存在结果**：同一 **完整** 文件名（含 `-Manifold-k128` 等）已存在时仍会跳过；重跑 Manifold 实验前请删除或改名该 Manifold 专用 `jsonl`（主 README 对原版同理）。

---

## 7. 超参数敏感度分析（k 维度）

目标：固定 `T`、`layer_idx`、`steer=CRFDSteer`，分析流形子空间维度 `k` 对不同任务关键指标的影响：

- TruthfulQA：**True × Info** 与 **Perplexity**
- Ultrafeedback：**RM Mean** 与 **RM Win-Rate vs NoSteer**
- RealToxicityPrompts：**Toxicity** 与 **Perplexity**

### 7.1 一键脚本（推荐）

已提供驱动脚本：

- `scripts/sensitivity/run_ode_manifold_k_sensitivity_truthfulqa.py`
- `scripts/sensitivity/run_ode_manifold_k_sensitivity_ultrafeedback.py`
- `scripts/sensitivity/run_ode_manifold_k_sensitivity_toxicity.py`

默认配置：

- `model=Llama3.1-8B-Base`
- `layer=13`
- `steer.T=5.0`
- `seed=42`
- `k ∈ {16, 32, 64, 128, 256, 512, 1024, 4096}`
- `use_manifold=true` + `use_manifold_retraction=true`

执行（仓库根目录）：

```bash
# TruthfulQA
uv run python scripts/sensitivity/run_ode_manifold_k_sensitivity_truthfulqa.py

# Ultrafeedback（可选指定 RM）
uv run python scripts/sensitivity/run_ode_manifold_k_sensitivity_ultrafeedback.py \
  --reward-model Skywork/Skywork-Reward-V2-Llama-3.1-8B

# RealToxicityPrompts
uv run python scripts/sensitivity/run_ode_manifold_k_sensitivity_toxicity.py
```

该命令按顺序完成：

1. 提取每个 `k` 的 `V_k`（若 `manifold_Vk_layer<L>_k<K>.pt` 已存在则跳过）。
2. 对每个 `k` 运行 `truthfulqa_generate.py`（若对应 raw jsonl 已存在则跳过）。
3. 运行对应任务的 `*_eval.py` 汇总评估。
4. 读取评估 CSV 并画双轴图：
   - TruthfulQA：左轴 `True * Info`（柱状），右轴 `Perplexity`（虚线）
   - Ultrafeedback：左轴 `RM Mean`（柱状），右轴 `RM Win-Rate vs NoSteer`（虚线）
   - RealToxicityPrompts：左轴 `Toxicity`（柱状），右轴 `Perplexity`（虚线）

### 7.2 只绘图 / 分步执行

只基于现有评估结果绘图：

```bash
uv run python scripts/sensitivity/run_ode_manifold_k_sensitivity_truthfulqa.py --plot-only
uv run python scripts/sensitivity/run_ode_manifold_k_sensitivity_ultrafeedback.py --plot-only
uv run python scripts/sensitivity/run_ode_manifold_k_sensitivity_toxicity.py --plot-only
```

仅跳过某阶段：

```bash
uv run python scripts/sensitivity/run_ode_manifold_k_sensitivity_truthfulqa.py --skip-extract
uv run python scripts/sensitivity/run_ode_manifold_k_sensitivity_truthfulqa.py --skip-generate
uv run python scripts/sensitivity/run_ode_manifold_k_sensitivity_truthfulqa.py --skip-eval

uv run python scripts/sensitivity/run_ode_manifold_k_sensitivity_ultrafeedback.py --skip-extract
uv run python scripts/sensitivity/run_ode_manifold_k_sensitivity_ultrafeedback.py --skip-generate
uv run python scripts/sensitivity/run_ode_manifold_k_sensitivity_ultrafeedback.py --skip-eval

uv run python scripts/sensitivity/run_ode_manifold_k_sensitivity_toxicity.py --skip-extract
uv run python scripts/sensitivity/run_ode_manifold_k_sensitivity_toxicity.py --skip-generate
uv run python scripts/sensitivity/run_ode_manifold_k_sensitivity_toxicity.py --skip-eval
```

自定义参数示例：

```bash
uv run python scripts/sensitivity/run_ode_manifold_k_sensitivity_truthfulqa.py \
  --model Llama3.1-8B-Base \
  --layer 13 \
  --steer-T 5.0 \
  --seed 42 \
  --ks 16,32,64,128,256,512,1024,4096
```

```bash
uv run python scripts/sensitivity/run_ode_manifold_k_sensitivity_ultrafeedback.py \
  --model Llama3.1-8B-Base \
  --layer 13 \
  --steer-T 5.0 \
  --seed 42 \
  --ks 16,32,64,128,256,512,1024,4096 \
  --reward-model Skywork/Skywork-Reward-V2-Llama-3.1-8B
```

```bash
uv run python scripts/sensitivity/run_ode_manifold_k_sensitivity_toxicity.py \
  --model Llama3.1-8B-Base \
  --layer 13 \
  --steer-T 5.0 \
  --seed 42 \
  --ks 16,32,64,128,256,512,1024,4096
```

### 7.3 输出位置

- `V_k`：`data/truthfulqa/activations/<model>/manifold_Vk_layer<L>_k<K>.pt`
- raw 生成：`results/truthfulqa/raw_outputs/<model>/*.jsonl`
- 评估 CSV：`results/truthfulqa/eval_results/stat_results/<model>-l<L>-TruthfulQA-seed<S>.csv`
- 敏感度图：`results/truthfulqa/sensitivity/manifold_k_crfdsteer_T<T>_l<L>_seed<S>.png`

- `V_k`：`data/ultrafeedback/activations/<model>/manifold_Vk_layer<L>_k<K>.pt`
- raw 生成：`results/ultrafeedback/raw_outputs/<model>/*.jsonl`
- 评估 CSV：`results/ultrafeedback/eval_results/stat_results/<model>-l<L>-UltrafeedbackBinarized-seed<S>.csv`
- 敏感度图：`results/ultrafeedback/sensitivity/manifold_k_crfdsteer_T<T>_l<L>_seed<S>.png`

- `V_k`：`data/toxicity/activations/<model>/manifold_Vk_layer<L>_k<K>.pt`
- raw 生成：`results/toxicity/raw_outputs/<model>/*.jsonl`
- 评估 CSV：`results/toxicity/eval_results/stat_results/<model>-l<L>-RealToxicityPrompts-seed<S>.csv`
- 敏感度图：`results/toxicity/sensitivity/manifold_k_crfdsteer_T<T>_l<L>_seed<S>.png`

### 7.4 说明

- `k=4096` 表示“请求全维”；实际仍受 SVD 有效秩 `r=min(N,d)` 限制，脚本会自动使用 `k_eff=min(k,r)`。
- 绘图时默认只统计 **多步 CRFDSteer + Manifold-k{k}（含 retraction）**，并排除 `-proj` 与 `CLF` 变体，避免混入非本实验条件。

---

## 8. 常见问题与排查

**Q：`extract_manifold.py` 报找不到 `*activations_layer{L}.pt`**  
A：确认 `--task` / `--activations_dir` 与 `-m`、`-l` 是否正确；并确认已运行对应任务的 `extract_activations`。

**Q：Hydra 报错 `Could not override 'steer.use_manifold'` / `Key 'use_manifold' is not in struct`**  
A：流形相关键在 **`steer.kwargs`** 下。请使用 `steer.kwargs.use_manifold=true`、`steer.kwargs.vk_path=...`、`steer.kwargs.manifold_dim=128`，不要使用 `steer.use_manifold`（顶层没有该字段）。

**Q：生成阶段报 `use_manifold=True requires V_k or vk_path`**  
A：在 Hydra 中设置 **`steer.kwargs.use_manifold=true`** 的同时必须提供可用的 **`steer.kwargs.vk_path`**（或由 `--use_manifold` + `--vk_path` / 配置中的 `kwargs.vk_path` 提供）。

**Q：层维数与 `V_k` 形状不一致**  
A：`V_k` 的行维 `d` 必须与当前模型、当前 `layer_idx` 的隐藏维一致；换模型或换层需重新提取 `V_k`。

**Q：`--use_manifold` 写了但 Hydra 报未知参数**  
A：请确认使用的是已修改过的 `*_generate.py`（仓库中已在 import 后调用 `strip_manifold_flags_from_argv()`）。若你自行复制了旧脚本，需同步该逻辑。

**Q：`steer=CAA`（或 RepE/ITI/LinAcT/MiMiC）时报 `Key 'use_manifold' is not in struct`**  
A：请确认你使用的是已更新后的 `confs/steer/*.yaml`（这些文件里的 `kwargs` 已包含 `use_manifold`、`use_manifold_retraction`、`vk_path`、`manifold_dim`）。若你本地仍是旧版本，请先同步代码或手动补齐这些键。

**Q：仅想保范数、不想子空间投影**  
A：当前实现中投影仅在 `self.V_k is not None` 时启用。若只配置 `use_manifold_retraction` 而不提供 `V_k`、且 `use_manifold=false`，可走「仅收回」路径（需自行确认是否满足你的实验设定；默认 YAML 中两者多为 false）。

---

## 9. 相关源码文件（便于二次开发）

| 文件 | 作用 |
|------|------|
| `data/extract_manifold.py` | 离线 SVD，写出 `manifold_Vk_layer{L}_k{K}.pt` |
| `scripts/sensitivity/run_ode_manifold_k_sensitivity_truthfulqa.py` | k 维度敏感度全流程（extract / generate / eval / plot） |
| `scripts/sensitivity/run_ode_manifold_k_sensitivity_ultrafeedback.py` | Ultrafeedback 的 k 敏感度全流程 |
| `scripts/sensitivity/run_ode_manifold_k_sensitivity_toxicity.py` | RealToxicityPrompts 的 k 敏感度全流程 |
| `src/odesteer/steer/_base_steer.py` | 通用流形后处理（向量场投影 / 输出收回） |
| `src/odesteer/steer/_manifold.py` | 投影、收回、`V_k` 截断与从路径加载 |
| `src/odesteer/steer/_ode_steer.py` | 多步 `odeint` + 流形逻辑 |
| `src/odesteer/steer/_step_ode_steer.py` | 单步欧拉 + 流形逻辑 |
| `src/odesteer/utils/manifold_cli.py` | `--use_manifold` 等与 Hydra 的衔接；`steer_name_with_manifold_suffix` 区分输出文件名 |
| `confs/steer/*.yaml` | 各 steer 的默认参数；流形键位（`use_manifold` 等）已扩展到多数 steer |
| `prompt.md` | 方法动机与与 LoRAM / Selective Steering 的类比说明 |

---

## 10. 引用

若你在论文或报告中同时引用本方法与上游 ODESteer 论文，请至少包含 README 中 **ODESteer（ICLR 2026）的 BibTeX**，并在正文中说明 ODE-Manifold 为基于该实现的推理期流形扩展；论文中请将主方法命名为 **CRFDSteer**（代码中 `steer=CRFDSteer`）。

---

如有新的实验设定（例如每层不同 `V_k`、或对投影向量再归一化），可在 `src/odesteer/steer/_manifold.py` 与 `vector_field` 处集中修改，并保持与本文档同步更新。
