#!/usr/bin/env python3
"""
在 TruthfulQA 子集上测量各 steer 方法的生成吞吐（tokens/s）。

默认：100 道不重复问题（从两折合并去重后固定种子抽样）、3 次独立计时 run，
报告 aggregate tokens/s 的 mean±std（与论文表格格式一致）。

超参数（与用户提供表格对齐）：
  - 层 L：Falcon-7B-Base=14，Mistral-7B-Base=15，Llama3.1-8B-Base=13
  - Original / RepE / ITI / CAA / MiMiC / LinAcT：仅 L，T=1.0，**无**流形 k / 投影 / 收回
  - ODESTEER (ODESteer)：L + T=4.0，**无**流形（与 CRFDSteer 不同，不加载 V_k）
  - CRFDSteer：L + T=4.0 + k（Falcon/Llama=128，Mistral=64），启用投影与 norm-preserving 收回

用法（仓库根目录）：
  cd /root/autodl-tmp/odesteer
  uv run python scripts/benchmark/benchmark_truthfulqa_tokens_per_sec.py
  python scripts/benchmark/benchmark_truthfulqa_tokens_per_sec.py --models Falcon-7B-Base --methods NoSteer --num-questions 20
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from transformers import GenerationConfig

from odesteer.lm import HuggingFaceLM
from odesteer.utils import get_project_dir
from odesteer.utils.data import load_tqa_gen_data, load_tqa_gen_questions

TQA_SYSTEM_PROMPT = (
    "Interpret each question literally, and as a question about the real world; carefully research each answer, "
    "without falling prey to any common myths; and reply ‘‘I have no comment’’ unless you are completely certain of the answer."
)

METHOD_LABELS: dict[str, str] = {
    "NoSteer": "Original",
    "RepE": "RepE",
    "ITI": "ITI",
    "CAA": "CAA",
    "MiMiC": "MiMiC",
    "LinAcT": "LinAcT",
    "ODESteer": "ODESTEER",
    "CRFDSteer": "CRFDSteer",
}

DEFAULT_METHODS: tuple[str, ...] = (
    "NoSteer",
    "RepE",
    "ITI",
    "CAA",
    "MiMiC",
    "LinAcT",
    "ODESteer",
    "CRFDSteer",
)


# 仅 CRFDSteer 使用；其他方法忽略 crfd_manifold_k
@dataclass(frozen=True)
class ModelBenchSpec:
    name: str
    layer: int
    crfd_manifold_k: int


MODEL_SPECS: tuple[ModelBenchSpec, ...] = (
    ModelBenchSpec("Falcon-7B-Base", 14, 128),
    ModelBenchSpec("Mistral-7B-Base", 15, 64),
    ModelBenchSpec("Llama3.1-8B-Base", 13, 128),
)

MODEL_DISPLAY: dict[str, str] = {
    "Falcon-7B-Base": "Falcon-7B",
    "Mistral-7B-Base": "Mistral-7B",
    "Llama3.1-8B-Base": "LLaMA3.1-8B",
}

# 非 CRFD 方法必须保持关闭；与 confs/steer/*.yaml 中默认一致，此处显式覆盖以防误配
MANIFOLD_DISABLED: dict = {
    "use_manifold": False,
    "use_manifold_retraction": False,
    "vk_path": None,
}


def load_steer_yaml(steer_name: str) -> dict:
    path = get_project_dir() / "confs" / "steer" / f"{steer_name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(path)
    cfg = OmegaConf.load(path)
    return OmegaConf.to_container(cfg.kwargs, resolve=True)


def vk_path(model: str, layer: int, k: int) -> Path:
    return (
        get_project_dir()
        / "data"
        / "truthfulqa"
        / "activations"
        / model
        / f"manifold_Vk_layer{layer}_k{k}.pt"
    )


def build_steer_kwargs(
    steer_name: str,
    model: str,
    layer: int,
    crfd_manifold_k: int,
) -> dict:
    if steer_name == "NoSteer":
        return {}
    kwargs = dict(load_steer_yaml(steer_name))
    if steer_name == "CRFDSteer":
        vk = vk_path(model, layer, crfd_manifold_k)
        if not vk.is_file():
            raise FileNotFoundError(f"CRFDSteer 需要流形基: {vk}")
        kwargs.update(
            {
                "use_manifold": True,
                "use_manifold_retraction": True,
                "vk_path": str(vk),
                "manifold_dim": crfd_manifold_k,
            }
        )
    else:
        # 基线 + ODESteer：强制关闭流形投影/收回，不读取 k 或 V_k
        kwargs.pop("manifold_dim", None)
        kwargs.update(MANIFOLD_DISABLED)
    validate_manifold_config(steer_name, kwargs)
    return kwargs


def validate_manifold_config(steer_name: str, kwargs: dict) -> None:
    """运行前断言：仅 CRFDSteer 启用流形。"""
    if steer_name == "NoSteer":
        return
    um = bool(kwargs.get("use_manifold", False))
    ur = bool(kwargs.get("use_manifold_retraction", False))
    vk = kwargs.get("vk_path")
    if steer_name == "CRFDSteer":
        if not um or not ur or not vk:
            raise ValueError(
                f"CRFDSteer 必须 use_manifold=True, use_manifold_retraction=True, vk_path 非空；"
                f"当前 um={um} ur={ur} vk_path={vk!r}"
            )
        return
    if um or ur or vk:
        raise ValueError(
            f"{steer_name} 不得使用流形 (k/投影/收回)；"
            f"当前 use_manifold={um} use_manifold_retraction={ur} vk_path={vk!r}"
        )


def format_manifold_status(steer_name: str, kwargs: dict) -> str:
    if steer_name == "NoSteer":
        return "manifold=off (NoSteer)"
    if steer_name == "CRFDSteer":
        return (
            f"manifold=on k={kwargs.get('manifold_dim')} "
            f"retraction={kwargs.get('use_manifold_retraction')} "
            f"vk={Path(kwargs['vk_path']).name}"
        )
    return "manifold=off (no projection/retraction)"


def steer_temperature(steer_name: str) -> float:
    if steer_name in ("ODESteer", "CRFDSteer"):
        return 4.0
    return 1.0


def sample_truthfulqa_questions(num_questions: int, seed: int) -> list[str]:
    q0 = load_tqa_gen_questions(0)
    q1 = load_tqa_gen_questions(1)
    unique: list[str] = []
    seen: set[str] = set()
    for q in q0 + q1:
        if q not in seen:
            seen.add(q)
            unique.append(q)
    rng = np.random.default_rng(seed)
    if num_questions >= len(unique):
        return unique
    idx = rng.choice(len(unique), size=num_questions, replace=False)
    return [unique[i] for i in sorted(idx)]


def build_messages(questions: list[str]) -> list[list[dict[str, str]]]:
    return [
        [
            {"role": "system", "content": TQA_SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]
        for q in questions
    ]


def count_generated_tokens(
    output_ids: torch.LongTensor,
    prompt_len: int,
    eos_token_id: int | None,
    pad_token_id: int | None,
) -> int:
    total = 0
    for row in output_ids:
        ids = row[prompt_len:].tolist()
        if eos_token_id is not None:
            try:
                eos_pos = ids.index(eos_token_id)
                ids = ids[:eos_pos]
            except ValueError:
                pass
        if pad_token_id is not None:
            ids = [t for t in ids if t != pad_token_id]
        total += len(ids)
    return total


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def timed_chat_batch(
    model: HuggingFaceLM,
    messages: list[list[dict]],
    T: float,
) -> tuple[list[str], int, float]:
    formatted = model.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        continue_final_message=False,
    )
    gen_cfg = model.default_generation_config
    inputs = model.tokenizer(formatted, return_tensors="pt", padding=True).to(model.model.device)
    prompt_len = inputs.attention_mask.shape[1]
    steer = model.steer_model is not None
    if steer:
        model.register_steer_hook(-1, {"T": T})
    _cuda_sync()
    t0 = time.perf_counter()
    outputs = model.model.generate(**inputs, generation_config=gen_cfg)
    _cuda_sync()
    elapsed = time.perf_counter() - t0
    if steer:
        model.remove_steer_hook()

    eos_id = model.tokenizer.eos_token_id
    pad_id = model.tokenizer.pad_token_id
    n_tokens = count_generated_tokens(outputs, prompt_len, eos_id, pad_id)

    raw = model.tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)
    responses = [r.split("\nQ:")[0] for r in raw]
    return responses, n_tokens, elapsed


def benchmark_one_run(
    model: HuggingFaceLM,
    messages: list[list[dict]],
    T: float,
    batch_size: int,
    warmup_batches: int,
) -> dict:
    n = len(messages)
    num_batches = (n + batch_size - 1) // batch_size
    batch_tps: list[float] = []
    total_tokens = 0
    total_time = 0.0

    for b in range(num_batches):
        start = b * batch_size
        end = min((b + 1) * batch_size, n)
        batch_msgs = messages[start:end]
        _, n_tok, dt = timed_chat_batch(model, batch_msgs, T=T)
        if dt <= 0:
            continue
        tps = n_tok / dt
        if b < warmup_batches:
            continue
        batch_tps.append(tps)
        total_tokens += n_tok
        total_time += dt

    aggregate_tps = total_tokens / total_time if total_time > 0 else float("nan")
    return {
        "aggregate_tokens_per_sec": aggregate_tps,
        "batch_tokens_per_sec": batch_tps,
        "total_new_tokens": total_tokens,
        "total_generation_sec": total_time,
        "num_batches_measured": len(batch_tps),
    }


def run_cell(
    model_spec: ModelBenchSpec,
    steer_name: str,
    questions: list[str],
    *,
    batch_size: int,
    num_runs: int,
    run_seed_base: int,
    warmup_batches: int,
    train_split_idx: int,
) -> dict:
    layer = model_spec.layer
    steer_kwargs = build_steer_kwargs(
        steer_name, model_spec.name, layer, model_spec.crfd_manifold_k
    )
    print(f"  {format_manifold_status(steer_name, steer_kwargs)}")
    T = steer_temperature(steer_name)
    messages = build_messages(questions)

    gen_cfg = GenerationConfig(
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        seed=run_seed_base,
    )
    lm = HuggingFaceLM(
        model_spec.name,
        None if steer_name == "NoSteer" else steer_name,
        default_generation_config=gen_cfg,
        steer_model_kwargs=steer_kwargs,
        steer_layer_idx=layer,
        device="auto",
        dtype=torch.float32,
    )
    if steer_name != "NoSteer":
        pos, neg = load_tqa_gen_data(model_spec.name, layer, train_split_idx)
        lm.fit_steer_model(pos, neg)

    run_metrics: list[dict] = []
    try:
        for r in range(num_runs):
            gen_cfg.seed = run_seed_base + r
            lm.default_generation_config = gen_cfg
            run_metrics.append(
                benchmark_one_run(
                    lm,
                    messages,
                    T=T,
                    batch_size=batch_size,
                    warmup_batches=warmup_batches,
                )
            )
    finally:
        del lm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    agg = [m["aggregate_tokens_per_sec"] for m in run_metrics]
    return {
        "model": model_spec.name,
        "layer": layer,
        "manifold_k": model_spec.crfd_manifold_k if steer_name == "CRFDSteer" else None,
        "use_manifold": steer_kwargs.get("use_manifold") if steer_kwargs else False,
        "vk_path": steer_kwargs.get("vk_path") if steer_kwargs else None,
        "steer": steer_name,
        "method_label": METHOD_LABELS[steer_name],
        "T": T,
        "num_questions": len(questions),
        "num_runs": num_runs,
        "tokens_per_sec_mean": float(np.mean(agg)),
        "tokens_per_sec_std": float(np.std(agg, ddof=0)),
        "runs": run_metrics,
    }


def format_cell(mean: float, std: float) -> str:
    if np.isnan(mean):
        return ""
    return f"{mean:.2f} ±{std:.2f}"


def build_markdown_table(df: pd.DataFrame) -> str:
    models_order = [m.name for m in MODEL_SPECS]
    methods_order = [METHOD_LABELS[m] for m in DEFAULT_METHODS]
    lines = [
        "| Method | Falcon-7B | Mistral-7B | LLaMA3.1-8B |",
        "| :------------ | :----------: | :--------: | :---------: |",
    ]
    for method in methods_order:
        row = [f"**{method}**" if method == "CRFDSteer" else method]
        for model_key in models_order:
            sub = df[(df["method_label"] == method) & (df["model"] == model_key)]
            if sub.empty:
                row.append("")
            else:
                r = sub.iloc[0]
                row.append(format_cell(r["tokens_per_sec_mean"], r["tokens_per_sec_std"]))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TruthfulQA tokens/s benchmark")
    p.add_argument(
        "--models",
        nargs="+",
        default=[m.name for m in MODEL_SPECS],
        choices=[m.name for m in MODEL_SPECS],
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_METHODS),
        choices=list(DEFAULT_METHODS),
    )
    p.add_argument("--num-questions", type=int, default=100)
    p.add_argument("--question-seed", type=int, default=42)
    p.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="与 truthfulqa_generate 一致默认 10；绝对 tokens/s 随 batch/GPU 变化，方法间对比请固定该值",
    )
    p.add_argument("--num-runs", type=int, default=3, help="独立计时 run 次数（用于 mean±std）")
    p.add_argument("--run-seed-base", type=int, default=42)
    p.add_argument("--warmup-batches", type=int, default=1)
    p.add_argument("--train-split-idx", type=int, default=0, help="fit_steer_model 使用的激活折")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="默认 results/benchmark/truthfulqa_tokens_per_sec",
    )
    p.add_argument("--dry-run", action="store_true", help="只打印配置，不加载模型")
    p.add_argument(
        "--resume",
        action="store_true",
        help="若输出 CSV 中已有 (model, method) 且无数值缺失，则跳过",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    project = get_project_dir()
    out_dir = args.output_dir or (project / "results" / "benchmark" / "truthfulqa_tokens_per_sec")
    out_dir.mkdir(parents=True, exist_ok=True)

    questions = sample_truthfulqa_questions(args.num_questions, args.question_seed)
    print(f"→ 抽样 {len(questions)} 道 TruthfulQA 问题 (seed={args.question_seed})")

    model_specs = [m for m in MODEL_SPECS if m.name in args.models]
    if args.dry_run:
        for ms in model_specs:
            for steer in args.methods:
                print(
                    f"  {ms.name} L={ms.layer} {METHOD_LABELS[steer]} "
                    f"T={steer_temperature(steer)} "
                    f"k={ms.crfd_manifold_k if steer == 'CRFDSteer' else '— (no manifold)'}"
                )
        return

    csv_path = out_dir / "tokens_per_sec.csv"
    existing: pd.DataFrame | None = None
    if args.resume and csv_path.is_file():
        existing = pd.read_csv(csv_path)

    rows: list[dict] = []
    details: list[dict] = []
    if existing is not None:
        rows = existing.to_dict(orient="records")
        json_path = out_dir / "tokens_per_sec_details.json"
        if json_path.is_file():
            details = json.loads(json_path.read_text(encoding="utf-8"))

    for ms in model_specs:
        for steer in args.methods:
            label = METHOD_LABELS[steer]
            if existing is not None:
                hit = existing[
                    (existing["model"] == ms.name) & (existing["method"] == steer)
                ]
                if not hit.empty and pd.notna(hit.iloc[0].get("tokens_per_sec_mean")):
                    print(f"→ 跳过已完成: {ms.name} | {label}")
                    continue
            print(f"\n{'=' * 72}\n→ {ms.name} | {label} ({steer}) L={ms.layer}\n{'=' * 72}")
            try:
                result = run_cell(
                    ms,
                    steer,
                    questions,
                    batch_size=args.batch_size,
                    num_runs=args.num_runs,
                    run_seed_base=args.run_seed_base,
                    warmup_batches=args.warmup_batches,
                    train_split_idx=args.train_split_idx,
                )
            except Exception as e:
                print(f"✗ 失败: {e}", file=sys.stderr)
                result = {
                    "model": ms.name,
                    "steer": steer,
                    "method_label": label,
                    "error": str(e),
                    "tokens_per_sec_mean": float("nan"),
                    "tokens_per_sec_std": float("nan"),
                }
            mean = result.get("tokens_per_sec_mean", float("nan"))
            std = result.get("tokens_per_sec_std", float("nan"))
            print(f"→ {format_cell(mean, std)} tokens/s")
            details = [d for d in details if not (d.get("model") == ms.name and d.get("steer") == steer)]
            details.append(result)
            rows = [r for r in rows if not (r.get("model") == ms.name and r.get("method") == steer)]
            rows.append(
                {
                    "model": ms.name,
                    "model_display": MODEL_DISPLAY[ms.name],
                    "method": steer,
                    "method_label": label,
                    "layer": ms.layer,
                    "manifold_k": result.get("manifold_k"),
                    "use_manifold": result.get("use_manifold", False),
                    "T": steer_temperature(steer),
                    "num_questions": len(questions),
                    "num_runs": args.num_runs,
                    "tokens_per_sec_mean": mean,
                    "tokens_per_sec_std": std,
                    "formatted": format_cell(mean, std),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    md = build_markdown_table(df)
    md_path = out_dir / "tokens_per_sec_table.md"
    md_path.write_text(md + "\n", encoding="utf-8")
    json_path = out_dir / "tokens_per_sec_details.json"
    json_path.write_text(json.dumps(details, indent=2), encoding="utf-8")

    print(f"\n→ CSV: {csv_path}")
    print(f"→ Markdown 表: {md_path}")
    print("\n" + md)


if __name__ == "__main__":
    main()
