# COPA Implementation Summary

## 修改了哪些文件

### 新增

| 文件 | 目的 |
|------|------|
| `src/odesteer/utils/copa.py` | COPA XML 加载、划分、提示词、正负元数据、泄漏检查 |
| `confs/copa.yaml` | Hydra 入口；锁定 `Llama3.1-8B-Base` / `layer_idx=13` / `seed=42` |
| `scripts/copa/copa_run.py` | 正式实验：提激活、fit、forced-choice 评分、指标与导出 |
| `scripts/copa/copa_smoke_test.py` | 无 GPU 的数据/注册表 smoke |
| `COPA_IMPLEMENTATION_SUMMARY.md` | 本说明 |

### 未修改

- 未改动 8 个 steer 实现、`HuggingFaceLM`、TruthfulQA/UltraFeedback/Toxicity 脚本或已有非 COPA 的 `results/` 目录。

## 为什么这样修改

在**不重构、不重调参**前提下，仅增加 COPA 数据适配与选择题评估，用于验证激活引导能否迁移到因果常识推理（**zero-tuning transfer**）。

## 复用的现有模块

- `HuggingFaceLM`（加载、`register_steer_prob_hook` / `fit_steer_model`、`_forward_for_activation_capture`）
- `get_steer_model` / 各 `confs/steer/*.yaml`
- 参数常量对齐：`scripts/repro/run_multi_seed_grid.py`（L=13，T=5/1，k=256 for TQA）
- TruthfulQA 预计算 `manifold_Vk_layer13_k256.pt`（GSCSteer OTSP，不在 COPA test 上估计）

## 数据集与划分

- **来源**：`data/COPA/datasets/copa-dev.xml` + `copa-test.xml`（COPA-resources v1.0）
- **标定 / 构造**：500 development（idx 1–500）
- **测试**：500 official test（idx 501–1000）
- **开发期可选**：development 按 `question×label` 分层为 build=400 / validation=100；**正式实验 fit 用满 500 development**
- 标定与 test 无 idx/文本泄漏

## 正负样本

对每条标定样本：

- 正：`format_copa_prompt(s) + gold_letter`（语义上）；实现上对 prompt / letter **分别 tokenize 再拼接 id**，避免 Llama 把 `Answer:`+`A` 合并成同一长度序列
- 负：同上，字母为 incorrect
- 提取位置：拼接序列的 **最后一个 token**（答案字母 token），与既有 EOS/尾 token 规则一致

## 选择题评分

**Forced-choice 条件对数似然**（非自由生成）：

- 同一 prompt 下分别对候选 `"A"` / `"B"` 求 token log-prob 之和（多 token 则求和）；另存 mean log-prob
- Tokenization：prompt 与 answer **分开编码再 concat**（修复 `prompt+"A"` 与 `prompt` 非前缀的问题）
- 全程启用对应方法的 `steer_prob_hook`（从答案起始位置起干预）；Original 不注册 Hook
- `score_A > score_B → A`；反之 B；相等固定选 A 并记 `is_tie`
- Invalid Output Rate = 0（不适用）

## 8 个方法与参数（zero-tuning）

| Display | 代码 | T | 其他 |
|---------|------|---|------|
| Original | `NoSteer` | — | 无 Hook |
| RepE / ITI / CAA | 同名 | 1.0 | layer 13 |
| MiMiC / Linear-AcT | `MiMiC` / `LinAcT` | 1.0 | 同上 |
| ODESteer | `ODESteer` | 5.0 | euler steps=10，**无**流形 |
| GSCSteer | `CRFDSteer` | 5.0 | `use_manifold` + retraction，`k=256`，Vk 来自 TQA |

- 模型：`Llama3.1-8B-Base` → `meta-llama/Llama-3.1-8B`（**Base，非 Instruct**）
- `layer_idx=13`（0 起；论文常写第 14 层）
- `dtype=float32`；`temperature=0.7 / top_p=0.9 / repetition_penalty=1.1` 写入配置快照；forced-choice **不采样**
- **未**在 COPA 上搜索层、T、k、步数等；结果标记 `truthfulqa_llama3.1-8b_zero_tuning_transfer`

## 如何运行

```bash
cd /root/autodl-tmp/odesteer
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# 数据 + 注册表 smoke（无需 GPU）
uv run python -u scripts/copa/copa_smoke_test.py

# 8 样本 × 8 方法端到端（加载模型）；输出到 results/copa/llama3.1-8b/smoke/
uv run python -u scripts/copa/copa_run.py mode=smoke

# 正式：500 cal + 500 test × 8 方法 → results/copa/llama3.1-8b/
uv run python -u scripts/copa/copa_run.py mode=full
```

## Smoke 结果（mode=smoke，n=8，非正式 Acc）

| Method | Correct/Total | Accuracy(%) |
|--------|---------------|-------------|
| Original | 5/8 | 62.5 |
| RepE | 5/8 | 62.5 |
| ITI | 5/8 | 62.5 |
| CAA | 5/8 | 62.5 |
| MiMiC | 6/8 | 75.0 |
| Linear-AcT | 5/8 | 62.5 |
| ODESteer | 7/8 | 87.5 |
| GSCSteer | 5/8 | 62.5 |

说明：以上仅为管道验证。

## 正式结果（mode=full，500 cal / 500 test，已执行）

| Method | Correct/Total | Accuracy(%) | ΔAcc(pp) | Cause | Effect | Choice-1 | Choice-2 | 95% CI |
|--------|---------------|-------------|----------|-------|--------|----------|----------|--------|
| Original | 338/500 | 67.6 | 0.0 | 62.4 | 72.8 | 35.2 | 100.0 | [63.40, 71.60] |
| RepE | 313/500 | 62.6 | -5.0 | 57.2 | 68.0 | 25.2 | 100.0 | [58.20, 66.60] |
| ITI | 355/500 | 71.0 | 3.4 | 66.0 | 76.0 | 42.4 | 99.6 | [67.00, 75.00] |
| CAA | 356/500 | 71.2 | 3.6 | 66.0 | 76.4 | 42.8 | 99.6 | [67.20, 75.20] |
| MiMiC | 362/500 | 72.4 | 4.8 | 67.6 | 77.2 | 46.0 | 98.8 | [68.40, 76.40] |
| Linear-AcT | 355/500 | 71.0 | 3.4 | 66.0 | 76.0 | 42.4 | 99.6 | [67.00, 75.00] |
| ODESteer | 437/500 | 87.4 | 19.8 | 84.8 | 90.0 | 86.8 | 88.0 | [84.40, 90.40] |
| GSCSteer | 370/500 | 74.0 | 6.4 | 70.0 | 78.0 | 49.6 | 98.4 | [70.00, 77.80] |

McNemar（配对）：GSCSteer vs Original p≈1.86e-7；vs 最佳基线 ODESteer p≈4.94e-10。参数来源标记为 `truthfulqa_llama3.1-8b_zero_tuning_transfer`（非 COPA 最优调参）。

## 结果目录

- 正式：`results/copa/llama3.1-8b/`
- Smoke：`results/copa/llama3.1-8b/smoke/`

至少包含：

- `copa_predictions.csv`
- `copa_metrics.json`
- `copa_results_table.csv`
- `copa_split_statistics.json`
- `copa_resolved_config.yaml`
- `copa_contrastive_metadata_*.json`

## 未解决问题 / 注意

- GSCSteer 依赖已有 TQA `V_k`；若缺少 `manifold_Vk_layer13_k256.pt`，需先对 TruthfulQA 激活跑 `data/extract_manifold.py -k 256`。
- 本仓库方法名为 **CRFDSteer**；COPA 结果表展示名为 **GSCSteer**（与任务 brief 一致）。
- 论文若写「第 14 层 / T=…」与代码不一致时，**以本仓库正式配置为准**（`layer_idx=13`，ODE/GSC `T=5.0`）。
- zero-tuning 下 ODESteer 在 COPA 上高于 GSCSteer；这是迁移结果，不是 COPA 上重搜超参后的结论。
