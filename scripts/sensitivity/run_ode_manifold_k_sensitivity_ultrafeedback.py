#!/usr/bin/env python3
"""
ODE-Manifold 子空间维度 k 的超参数敏感度（Ultrafeedback + CRFDSteer + Llama3.1-8B-Base 默认）。

流程：
  1) 对每个 k 调用 data/extract_manifold.py（若对应 .pt 已存在则跳过）
  2) 对每个随机种子、每个 k 调用 scripts/ultrafeedback/ultrafeedback_generate.py（若目标 jsonl 已存在则跳过）
  3) 每个种子调用 scripts/ultrafeedback/ultrafeedback_eval.py 汇总评估
  4) 从各 seed 的 eval CSV 聚合 CRFDSteer+Manifold 各行，按 k 计算跨种子均值与样本标准差，
     并对均值做 bootstrap 95% CI，**仅绘制 RM Win-Rate vs NoSteer**；Ultrafeedback 单独出图。

用法（在仓库根目录）：
  uv run python scripts/sensitivity/run_ode_manifold_k_sensitivity_ultrafeedback.py
  uv run python scripts/sensitivity/run_ode_manifold_k_sensitivity_ultrafeedback.py --plot-only
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

from manifold_k_seed_aggregate import aggregate_metrics_by_k, parse_seeds


def run_cmd(cmd: list[str], cwd: Path, env: dict | None = None) -> None:
    print("\n→ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=cwd, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {r.returncode}")


def expected_raw_jsonl(
    project_dir: Path, model: str, layer: int, k: int, seed: int
) -> Path:
    out_dir = project_dir / "results" / "ultrafeedback" / "raw_outputs" / model
    if not out_dir.is_dir():
        return out_dir / f"{model}-l{layer}-CRFDSteer-Manifold-k{k}-UltrafeedbackBinarized-seed{seed}.jsonl"
    pat = f"*-Manifold-k{k}-UltrafeedbackBinarized-seed{seed}.jsonl"
    matches = sorted(out_dir.glob(pat))
    if matches:
        return matches[0]
    return out_dir / f"{model}-l{layer}-CRFDSteer-Manifold-k{k}-UltrafeedbackBinarized-seed{seed}.jsonl"


def vk_path_for_k(project_dir: Path, model: str, layer: int, k: int) -> Path:
    return (
        project_dir
        / "data"
        / "ultrafeedback"
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
        "ultrafeedback",
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
        "scripts/ultrafeedback/ultrafeedback_generate.py",
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


def step_eval(
    project_dir: Path, model: str, layer: int, seed: int, reward_model: str, uv: str, skip: bool
) -> None:
    if skip:
        print("[skip-eval]")
        return
    cmd = [
        uv,
        "run",
        "python",
        "-u",
        "scripts/ultrafeedback/ultrafeedback_eval.py",
        "-m",
        model,
        "-l",
        str(layer),
        "-s",
        str(seed),
        "-r",
        reward_model,
        "-d",
    ]
    run_cmd(cmd, cwd=project_dir)


def load_manifold_k_rows(
    project_dir: Path, model: str, layer: int, seed: int
) -> pd.DataFrame:
    csv_path = (
        project_dir
        / "results"
        / "ultrafeedback"
        / "eval_results"
        / "stat_results"
        / f"{model}-l{layer}-UltrafeedbackBinarized-seed{seed}.csv"
    )
    if not csv_path.is_file():
        raise FileNotFoundError(f"Eval CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    model_key = f"{model}-l{layer}"
    sub = df[df["Model"] == model_key].copy()
    sm = sub["Steering Method"].astype(str)
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
    seeds: tuple[int, ...],
    steer_T: float,
    *,
    n_boot: int,
    bootstrap_seed: int,
    ci: float,
    outfile: Path | None = None,
) -> Path:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError as e:
        raise RuntimeError("matplotlib is required for plotting") from e

    wr_col = "RM Win-Rate vs NoSteer"
    seed_frames: list[pd.DataFrame] = []
    for seed in seeds:
        sub = load_manifold_k_rows(project_dir, model, layer, seed)
        if sub.empty:
            raise RuntimeError(
                f"No CRFDSteer/ODESteer + Manifold-k rows for seed={seed}. Run eval after generations."
            )
        seed_frames.append(sub)

    agg = aggregate_metrics_by_k(
        seed_frames,
        [wr_col],
        k_col="_k",
        n_boot=n_boot,
        ci=ci,
        bootstrap_seed=bootstrap_seed,
    )

    x = agg["_k"].astype(int).tolist()
    positions = list(range(len(x)))
    color_wr = "#ff7f0e"

    fig, ax1 = plt.subplots(figsize=(10, 5))

    wr_mean = f"{wr_col}_mean"
    wr_lo = f"{wr_col}_ci_lo"
    wr_hi = f"{wr_col}_ci_hi"

    ax1.plot(
        positions,
        agg[wr_mean].tolist(),
        color=color_wr,
        linestyle="-",
        marker="o",
        linewidth=2,
        markersize=6,
        label="Win-Rate (mean)",
    )
    ax1.fill_between(
        positions,
        agg[wr_lo].tolist(),
        agg[wr_hi].tolist(),
        color=color_wr,
        alpha=0.22,
        linewidth=0,
        label="Win-Rate bootstrap 95% CI",
    )
    ax1.set_xlabel("Manifold subspace dimension $k$")
    ax1.set_ylabel("RM Win-Rate vs NoSteer")
    ax1.set_xticks(positions)
    ax1.set_xticklabels([str(v) for v in x], rotation=45, ha="right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.legend(loc="best", frameon=True)

    seed_tag = "-".join(str(s) for s in seeds)
    fig.suptitle(
        f"{model} | Ultrafeedback | CRFDSteer + Manifold | "
        f"layer={layer}, T={steer_T}, seeds={seed_tag} | Win-Rate: mean ± bootstrap {int(ci * 100)}% CI"
    )

    fig.tight_layout(rect=[0, 0.06, 1, 0.96])

    out = outfile or (
        project_dir
        / "results"
        / "ultrafeedback"
        / "sensitivity"
        / f"manifold_k_crfdsteer_T{steer_T}_l{layer}_seeds{seed_tag}_bootci.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    agg_csv = out.with_name(out.stem + "_aggregate.csv")
    agg.to_csv(agg_csv, index=False)
    print(f"✓ Saved figure: {out}")
    print(f"✓ Saved aggregate (mean/std across seeds + bootstrap CI): {agg_csv}")
    return out


def parse_ks(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ODE-Manifold k sweep for Ultrafeedback + CRFDSteer (extract → generate → eval → plot)"
    )
    parser.add_argument("--model", type=str, default="Llama3.1-8B-Base")
    parser.add_argument("--layer", type=int, default=13)
    parser.add_argument("--steer-T", type=float, default=5.0, dest="steer_T")
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,123,456",
        help="Comma-separated random seeds (generate + eval per seed)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="If set, overrides --seeds to a single seed (backward compatibility)",
    )
    parser.add_argument(
        "--reward-model",
        type=str,
        default="Skywork/Skywork-Reward-V2-Llama-3.1-8B",
        help="Reward model for ultrafeedback_eval.py",
    )
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
    parser.add_argument("--bootstrap-n", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--uv", type=str, default="uv", help="uv executable name")
    args = parser.parse_args()

    project_dir = get_project_dir()
    os.chdir(project_dir)
    ks = parse_ks(args.ks)
    out_fig = Path(args.output_figure) if args.output_figure else None
    seeds = (args.seed,) if args.seed is not None else parse_seeds(args.seeds)

    if args.plot_only:
        plot_sensitivity(
            project_dir,
            args.model,
            args.layer,
            seeds,
            args.steer_T,
            n_boot=args.bootstrap_n,
            bootstrap_seed=args.bootstrap_seed,
            ci=args.ci,
            outfile=out_fig,
        )
        return

    for seed in seeds:
        print(f"\n######## seed = {seed} ########")
        for k in ks:
            print(f"\n=== k = {k} ===")
            step_extract(project_dir, args.model, args.layer, k, args.uv, args.skip_extract)
            step_generate(
                project_dir,
                args.model,
                args.layer,
                k,
                args.steer_T,
                seed,
                args.uv,
                args.skip_generate,
            )

        step_eval(
            project_dir,
            args.model,
            args.layer,
            seed,
            args.reward_model,
            args.uv,
            args.skip_eval,
        )

    plot_sensitivity(
        project_dir,
        args.model,
        args.layer,
        seeds,
        args.steer_T,
        n_boot=args.bootstrap_n,
        bootstrap_seed=args.bootstrap_seed,
        ci=args.ci,
        outfile=out_fig,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
