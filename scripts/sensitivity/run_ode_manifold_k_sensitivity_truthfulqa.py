#!/usr/bin/env python3
"""
ODE-Manifold 子空间维度 k 的超参数敏感度（TruthfulQA + CRFDSteer + Llama3.1-8B-Base 默认）。

流程：
  1) 对每个 k 调用 data/extract_manifold.py（若对应 .pt 已存在则跳过）
  2) 对每个 k 调用 scripts/truthfulqa/truthfulqa_generate.py（若目标 jsonl 已存在则跳过）
  3) 调用 scripts/truthfulqa/truthfulqa_eval.py 汇总评估
  4) 从 eval CSV 读取 CRFDSteer+Manifold 各行（兼容旧 CSV 中的 ODESteer 前缀），绘制 True×Info（左轴）与 PPL（右轴）

用法（在仓库根目录）：
  uv run python scripts/sensitivity/run_ode_manifold_k_sensitivity_truthfulqa.py
  uv run python scripts/sensitivity/run_ode_manifold_k_sensitivity_truthfulqa.py --plot-only
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

from odesteer.utils import get_project_dir


def run_cmd(cmd: list[str], cwd: Path, env: dict | None = None) -> None:
    print("\n→ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=cwd, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {r.returncode}")


def expected_raw_jsonl(
    project_dir: Path, model: str, layer: int, k: int, seed: int
) -> Path:
    """匹配已生成的 raw jsonl（steer_run_name 中含 ``-Manifold-k{k}-``）。"""
    out_dir = project_dir / "results" / "truthfulqa" / "raw_outputs" / model
    if not out_dir.is_dir():
        return out_dir / f"{model}-l{layer}-CRFDSteer-Manifold-k{k}-TruthfulQA-seed{seed}.jsonl"
    pat = f"*-Manifold-k{k}-TruthfulQA-seed{seed}.jsonl"
    matches = sorted(out_dir.glob(pat))
    if matches:
        return matches[0]
    return out_dir / f"{model}-l{layer}-CRFDSteer-Manifold-k{k}-TruthfulQA-seed{seed}.jsonl"


def vk_path_for_k(project_dir: Path, model: str, layer: int, k: int) -> Path:
    return (
        project_dir
        / "data"
        / "truthfulqa"
        / "activations"
        / model
        / f"manifold_Vk_layer{layer}_k{k}.pt"
    )


def step_extract(
    project_dir: Path,
    model: str,
    layer: int,
    k: int,
    uv: str,
    skip: bool,
) -> None:
    out = vk_path_for_k(project_dir, model, layer, k)
    if skip:
        print(f"  [skip-extract] k={k}")
        return
    if out.is_file() and out.stat().st_size > 0:
        print(f"  ✓ V_k exists, skip extract: {out}")
        return
    cmd = [
        uv,
        "run",
        "python",
        "data/extract_manifold.py",
        "-m",
        model,
        "-l",
        str(layer),
        "-k",
        str(k),
        "--task",
        "truthfulqa",
    ]
    run_cmd(cmd, cwd=project_dir)


def step_generate(
    project_dir: Path,
    model: str,
    layer: int,
    k: int,
    steer_T: float,
    seed: int,
    uv: str,
    skip: bool,
) -> None:
    if skip:
        print(f"  [skip-generate] k={k}")
        return
    raw = expected_raw_jsonl(project_dir, model, layer, k, seed)
    if raw.is_file():
        print(f"  ✓ Raw jsonl exists, skip generate: {raw.name}")
        return
    vk = vk_path_for_k(project_dir, model, layer, k)
    if not vk.is_file():
        raise FileNotFoundError(f"Missing V_k file for k={k}: {vk}")

    cmd = [
        uv,
        "run",
        "python",
        "-u",
        "scripts/truthfulqa/truthfulqa_generate.py",
        f"model={model}",
        f"layer_idx={layer}",
        "steer=CRFDSteer",
        f"steer.T={steer_T}",
        "steer.kwargs.use_manifold=true",
        "steer.kwargs.use_manifold_retraction=true",
        f"steer.kwargs.vk_path={vk.resolve()}",
        f"steer.kwargs.manifold_dim={k}",
        f"seed={seed}",
    ]
    run_cmd(cmd, cwd=project_dir)


def step_eval(project_dir: Path, model: str, layer: int, seed: int, uv: str, skip: bool) -> None:
    if skip:
        print("[skip-eval]")
        return
    cmd = [
        uv,
        "run",
        "python",
        "-u",
        "scripts/truthfulqa/truthfulqa_eval.py",
        "-m",
        model,
        "-l",
        str(layer),
        "-s",
        str(seed),
        "-d",
    ]
    run_cmd(cmd, cwd=project_dir)


def load_manifold_k_rows(
    project_dir: Path, model: str, layer: int, seed: int
) -> pd.DataFrame:
    csv_path = (
        project_dir
        / "results"
        / "truthfulqa"
        / "eval_results"
        / "stat_results"
        / f"{model}-l{layer}-TruthfulQA-seed{seed}.csv"
    )
    if not csv_path.is_file():
        raise FileNotFoundError(f"Eval CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    model_key = f"{model}-l{layer}"
    sub = df[df["Model"] == model_key].copy()
    sm = sub["Steering Method"].astype(str)
    # 仅多步 CRFDSteer（或旧 CSV 中的 ODESteer）+ 流形（带 IRR），排除 CLF 融合、仅投影 -proj、Step*
    sub = sub[
        (sm.str.startswith("CRFDSteer", na=False) | sm.str.startswith("ODESteer", na=False))
        & sm.str.contains("Manifold-k", na=False)
        & ~sm.str.contains("CLF", na=False)
        & ~sm.str.contains("-proj", na=False)
    ]
    pat = re.compile(r"Manifold-k(\d+)")
    ks: list[int] = []
    for s in sub["Steering Method"]:
        m = pat.search(str(s))
        ks.append(int(m.group(1)) if m else -1)
    sub["_k"] = ks
    sub = sub[sub["_k"] >= 0].drop_duplicates(subset=["_k"], keep="last").sort_values("_k")
    return sub


def plot_sensitivity(
    project_dir: Path,
    model: str,
    layer: int,
    seed: int,
    steer_T: float,
    outfile: Path | None = None,
) -> Path:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError as e:
        raise RuntimeError("matplotlib is required for plotting") from e

    sub = load_manifold_k_rows(project_dir, model, layer, seed)
    if sub.empty:
        raise RuntimeError(
            "No CRFDSteer/ODESteer + Manifold-k rows in eval CSV. Run eval after all generations."
        )

    x = sub["_k"].astype(int).tolist()
    y_ti = sub["True * Info"].astype(float).tolist()
    y_ppl = sub["Perplexity"].astype(float).tolist()

    fig, ax1 = plt.subplots(figsize=(10, 5))
    color_ti = "#1f77b4"
    color_ppl = "#ff7f0e"

    bars = ax1.bar(
        range(len(x)),
        y_ti,
        color=color_ti,
        alpha=0.85,
        label="True × Info",
        width=0.65,
    )
    ax1.set_xlabel("Manifold subspace dimension $k$")
    ax1.set_ylabel("True × Info", color=color_ti)
    ax1.tick_params(axis="y", labelcolor=color_ti)
    ax1.set_xticks(range(len(x)))
    ax1.set_xticklabels([str(v) for v in x], rotation=45, ha="right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = ax1.twinx()
    ax2.plot(
        range(len(x)),
        y_ppl,
        color=color_ppl,
        linestyle="--",
        marker="o",
        linewidth=2,
        markersize=6,
        label="Perplexity (PPL)",
    )
    ax2.set_ylabel("Perplexity (PPL)", color=color_ppl)
    ax2.tick_params(axis="y", labelcolor=color_ppl)

    fig.suptitle(
        f"{model} | TruthfulQA | CRFDSteer + Manifold | layer={layer}, T={steer_T}, seed={seed}"
    )

    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        lab1 + lab2,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        ncol=2,
        frameon=True,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])

    out = outfile or (
        project_dir
        / "results"
        / "truthfulqa"
        / "sensitivity"
        / f"manifold_k_crfdsteer_T{steer_T}_l{layer}_seed{seed}.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved figure: {out}")
    return out


def parse_ks(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ODE-Manifold k sweep for TruthfulQA + CRFDSteer (extract → generate → eval → plot)"
    )
    parser.add_argument("--model", type=str, default="Llama3.1-8B-Base")
    parser.add_argument("--layer", type=int, default=13)
    parser.add_argument("--steer-T", type=float, default=5.0, dest="steer_T")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ks",
        type=str,
        default="16,32,64,128,256,512,1024,4096",
        help="Comma-separated k list (4096 = request up to full rank; SVD may yield smaller k_eff)",
    )
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only load eval CSV and save figure (no extract/generate/eval)",
    )
    parser.add_argument("--output-figure", type=str, default=None, help="PNG path for plot")
    parser.add_argument("--uv", type=str, default="uv", help="uv executable name")
    args = parser.parse_args()

    project_dir = get_project_dir()
    os.chdir(project_dir)
    ks = parse_ks(args.ks)
    out_fig = Path(args.output_figure) if args.output_figure else None

    if args.plot_only:
        plot_sensitivity(
            project_dir,
            args.model,
            args.layer,
            args.seed,
            args.steer_T,
            outfile=out_fig,
        )
        return

    for k in ks:
        print(f"\n=== k = {k} ===")
        step_extract(project_dir, args.model, args.layer, k, args.uv, args.skip_extract)
        step_generate(
            project_dir,
            args.model,
            args.layer,
            k,
            args.steer_T,
            args.seed,
            args.uv,
            args.skip_generate,
        )

    step_eval(project_dir, args.model, args.layer, args.seed, args.uv, args.skip_eval)

    plot_sensitivity(
        project_dir,
        args.model,
        args.layer,
        args.seed,
        args.steer_T,
        outfile=out_fig,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
