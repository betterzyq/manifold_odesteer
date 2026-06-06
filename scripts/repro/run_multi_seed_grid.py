#!/usr/bin/env python3
"""
在多个随机种子下复现实验网格：4 个模型 × 3 个数据集 × 8 种 steer，并对各指标汇总 mean±std。

默认 8 种方法：NoSteer, RepE, ITI, CAA, MiMiC, LinAcT, CRFDSteer, ODESteer。
默认 3 个种子：42, 123, 456（可用 --seeds 覆盖）。

超参数（与论文/实验设定对齐，写死在脚本中，可用 JSON 覆盖）：
- 层 L：Llama/Qwen 13，Falcon 14，Mistral 15。
- 流形子空间维度 k（按任务）：见 MANIFOLD_K；CRFDSteer 会启用
  ``use_manifold`` + ``use_manifold_retraction``，并指向对应 ``manifold_Vk_layer{L}_k{k}.pt``。
- CRFDSteer / ODESteer 的 T：Llama 5、Falcon 22、Mistral 4、Qwen 16；
  Qwen 在 detoxification（toxicity）任务上 T=48。

用法（仓库根目录）：
  uv run python scripts/repro/run_multi_seed_grid.py
  uv run python scripts/repro/run_multi_seed_grid.py --aggregate-only
  uv run python scripts/repro/run_multi_seed_grid.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from odesteer.utils import get_project_dir


@dataclass(frozen=True)
class ModelSpec:
    name: str
    layer: int


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("Llama3.1-8B-Base", 13),
    ModelSpec("Falcon-7B-Base", 14),
    ModelSpec("Mistral-7B-Base", 15),
    ModelSpec("Qwen2.5-7B-Base", 13),
)

# (model, dataset_key) -> 流形维度 k；dataset_key: ultrafeedback / truthfulqa / toxicity
MANIFOLD_K: dict[tuple[str, str], int] = {
    ("Llama3.1-8B-Base", "ultrafeedback"): 32,
    ("Llama3.1-8B-Base", "truthfulqa"): 256,
    ("Llama3.1-8B-Base", "toxicity"): 128,
    ("Falcon-7B-Base", "ultrafeedback"): 128,
    ("Falcon-7B-Base", "truthfulqa"): 128,
    ("Falcon-7B-Base", "toxicity"): 128,
    ("Mistral-7B-Base", "ultrafeedback"): 32,
    ("Mistral-7B-Base", "truthfulqa"): 64,
    ("Mistral-7B-Base", "toxicity"): 128,
    ("Qwen2.5-7B-Base", "ultrafeedback"): 16,
    ("Qwen2.5-7B-Base", "truthfulqa"): 128,
    ("Qwen2.5-7B-Base", "toxicity"): 256,
}

# CRFDSteer / ODESteer 的 T（Qwen 在 toxicity 上为 48）
BASE_CRFD_T: dict[str, float] = {
    "Llama3.1-8B-Base": 5.0,
    "Falcon-7B-Base": 22.0,
    "Mistral-7B-Base": 4.0,
    "Qwen2.5-7B-Base": 16.0,
}

DEFAULT_SEEDS: tuple[int, ...] = (42, 123, 456)

# 8 种方法：(hydra steer 名, 是否设置 steer.T, T 值或 None 表示按模型/任务解析)
METHODS: tuple[tuple[str, bool, float | None], ...] = (
    ("NoSteer", False, None),
    ("RepE", True, 1.0),
    ("ITI", True, 1.0),
    ("CAA", True, 1.0),
    ("MiMiC", True, 1.0),
    ("LinAcT", True, 1.0),
    ("CRFDSteer", True, None),
    ("ODESteer", True, None),  # 与 CRFDSteer 共用同一 T 表（见 get_crfd_t）
)


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    generate_script: str
    eval_script: str
    csv_middle: str
    reward_model: str | None = None


DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        key="truthfulqa",
        generate_script="scripts/truthfulqa/truthfulqa_generate.py",
        eval_script="scripts/truthfulqa/truthfulqa_eval.py",
        csv_middle="TruthfulQA",
    ),
    DatasetSpec(
        key="ultrafeedback",
        generate_script="scripts/ultrafeedback/ultrafeedback_generate.py",
        eval_script="scripts/ultrafeedback/ultrafeedback_eval.py",
        csv_middle="UltrafeedbackBinarized",
        reward_model="Skywork/Skywork-Reward-V2-Llama-3.1-8B",
    ),
    DatasetSpec(
        key="toxicity",
        generate_script="scripts/toxicity/detox_generate.py",
        eval_script="scripts/toxicity/detox_eval.py",
        csv_middle="RealToxicityPrompts",
    ),
)


def manifold_k_for(model: str, ds_key: str, k_overrides: dict[str, int]) -> int:
    sk = f"{model}__{ds_key}"
    if sk in k_overrides:
        return int(k_overrides[sk])
    return MANIFOLD_K[(model, ds_key)]


def vk_path_for(
    project_dir: Path, model: str, layer: int, k: int, ds_key: str
) -> Path:
    return (
        project_dir
        / "data"
        / ds_key
        / "activations"
        / model
        / f"manifold_Vk_layer{layer}_k{k}.pt"
    )


def parse_json_dict(raw: str | None) -> dict[str, float | int]:
    if not raw:
        return {}
    d = json.loads(raw)
    return {str(k): v for k, v in d.items()}


def get_crfd_t(model: str, ds_key: str, t_overrides: dict[str, float]) -> float:
    """T 表 + Qwen detox=48；覆盖键可为 ``Model`` 或 ``Model__dataset_key``。"""
    sk = f"{model}__{ds_key}"
    if sk in t_overrides:
        return float(t_overrides[sk])
    if model in t_overrides:
        return float(t_overrides[model])
    if model == "Qwen2.5-7B-Base" and ds_key == "toxicity":
        return 48.0
    return float(BASE_CRFD_T[model])


def run_cmd(cmd: list[str], cwd: Path, dry_run: bool) -> None:
    print("\n→ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {r.returncode}")


def ensure_manifold_vk(
    project_dir: Path,
    uv: str,
    model: str,
    layer: int,
    k: int,
    extract_task: str,
    dry_run: bool,
) -> None:
    """若 ``manifold_Vk_layer{L}_k{k}.pt`` 不存在则调用 ``data/extract_manifold.py``。"""
    out = vk_path_for(project_dir, model, layer, k, extract_task)
    if out.is_file() and out.stat().st_size > 0:
        print(f"  ✓ V_k exists: {out.relative_to(project_dir)}", flush=True)
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
        extract_task,
    ]
    run_cmd(cmd, cwd=project_dir, dry_run=dry_run)


def per_seed_csv_path(
    project_dir: Path, ds: DatasetSpec, model: str, layer: int, seed: int
) -> Path:
    rel = (
        f"results/{ds.key}/eval_results/stat_results/"
        f"{model}-l{layer}-{ds.csv_middle}-seed{seed}.csv"
    )
    return project_dir / rel


def aggregate_mean_std(
    paths: list[Path],
    ddof: int = 1,
    *,
    model_label: str,
) -> pd.DataFrame:
    """按 Steering Method 对齐，对数值列计算 mean/std（跨 seed）。"""
    dfs: list[pd.DataFrame] = []
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(f"Missing eval CSV: {p}")
        dfs.append(pd.read_csv(p))

    n_seeds = len(dfs)
    std_ddof = 0 if n_seeds < 2 else ddof

    key = "Steering Method"
    all_methods = sorted(
        set().union(*[set(df[key].astype(str)) for df in dfs]),
        key=lambda s: s,
    )

    base = pd.DataFrame({"Steering Method": all_methods})
    out = base.copy()
    out.insert(0, "Model", model_label)

    numeric_cols: set[str] = set()
    for df in dfs:
        for c in df.columns:
            if c in ("Model", key):
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                numeric_cols.add(c)

    for col in sorted(numeric_cols):
        mat = []
        for df in dfs:
            m = df.set_index(key)[col].reindex(all_methods)
            mat.append(m.to_numpy(dtype=float))
        arr = np.vstack(mat).T
        out[f"{col}_mean"] = np.nanmean(arr, axis=1)
        out[f"{col}_std"] = np.nanstd(arr, axis=1, ddof=std_ddof)
        mean_c = f"{col}_mean"
        std_c = f"{col}_std"
        out[f"{col}_mean_pm_std"] = [
            f"{float(m):.6g}±{float(s):.6g}"
            for m, s in zip(out[mean_c].to_numpy(), out[std_c].to_numpy())
        ]

    return out


def build_generate_cmd(
    uv: str,
    ds: DatasetSpec,
    model: str,
    layer: int,
    steer: str,
    steer_T: float | None,
    seed: int,
    *,
    vk_path: Path | None = None,
    manifold_dim: int | None = None,
) -> list[str]:
    cmd = [
        uv,
        "run",
        "python",
        "-u",
        ds.generate_script,
        f"model={model}",
        f"layer_idx={layer}",
        f"steer={steer}",
        f"seed={seed}",
    ]
    if steer_T is not None:
        cmd.append(f"steer.T={steer_T}")
    if (
        steer == "CRFDSteer"
        and vk_path is not None
        and manifold_dim is not None
    ):
        cmd.extend(
            [
                "steer.kwargs.use_manifold=true",
                "steer.kwargs.use_manifold_retraction=true",
                f"steer.kwargs.vk_path={vk_path.resolve()}",
                f"steer.kwargs.manifold_dim={manifold_dim}",
            ]
        )
    return cmd


def build_eval_cmd(
    uv: str,
    ds: DatasetSpec,
    model: str,
    layer: int,
    seed: int,
    reward_override: str | None = None,
) -> list[str]:
    cmd = [
        uv,
        "run",
        "python",
        "-u",
        ds.eval_script,
        "-m",
        model,
        "-l",
        str(layer),
        "-s",
        str(seed),
        "-d",
    ]
    if ds.key == "ultrafeedback":
        rm = reward_override or ds.reward_model
        if rm:
            cmd.extend(["-r", rm])
    return cmd


def resolve_steer_t(
    steer: str,
    use_t: bool,
    t_val: float | None,
    model: str,
    ds: DatasetSpec,
    t_overrides: dict[str, float],
) -> float | None:
    if not use_t:
        return None
    if steer in ("CRFDSteer", "ODESteer"):
        return get_crfd_t(model, ds.key, t_overrides)
    return float(t_val) if t_val is not None else None


def parse_seeds(s: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-seed grid experiments (4 models × 3 datasets × 8 methods)"
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="Comma-separated seeds, e.g. 42,123,456",
    )
    parser.add_argument("--uv", type=str, default="uv")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Only aggregate existing per-seed CSVs into mean±std tables",
    )
    parser.add_argument(
        "--std-ddof",
        type=int,
        default=1,
        help="ddof for std (1=sample std with n seeds; 0=population std)",
    )
    parser.add_argument(
        "--crfd-t-json",
        type=str,
        default=None,
        help=(
            'JSON overrides for CRFDSteer/ODESteer T. Keys: ``"Model"`` 或 '
            '``"Model__dataset_key"``（如 ``"Qwen2.5-7B-Base__toxicity"``）。'
        ),
    )
    parser.add_argument(
        "--manifold-k-json",
        type=str,
        default=None,
        help=(
            'JSON overrides for manifold k. Keys: ``"Model__dataset_key"``，如 '
            '``"Llama3.1-8B-Base__truthfulqa": 128``。'
        ),
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated subset of model short names (default: all four)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated subset of: truthfulqa,ultrafeedback,toxicity",
    )
    parser.add_argument(
        "--reward-model",
        type=str,
        default=None,
        help="Ultrafeedback eval only: override -r reward model",
    )
    parser.add_argument(
        "--skip-ensure-vk",
        action="store_true",
        help="Do not run data/extract_manifold.py when V_k is missing (fail at generate if absent)",
    )
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    project_dir = get_project_dir()
    os.chdir(project_dir)

    t_overrides: dict[str, float] = {}
    for k, v in parse_json_dict(args.crfd_t_json).items():
        t_overrides[k] = float(v)

    k_overrides: dict[str, int] = {}
    for k, v in parse_json_dict(args.manifold_k_json).items():
        k_overrides[k] = int(v)

    model_specs = list(MODELS)
    if args.models:
        want = {x.strip() for x in args.models.split(",") if x.strip()}
        model_specs = [ms for ms in MODELS if ms.name in want]
        missing = want - {ms.name for ms in model_specs}
        if missing:
            raise ValueError(f"Unknown --models entries: {sorted(missing)}")

    dataset_specs = list(DATASETS)
    if args.datasets:
        want_d = {x.strip() for x in args.datasets.split(",") if x.strip()}
        dataset_specs = [ds for ds in DATASETS if ds.key in want_d]
        missing_d = want_d - {ds.key for ds in dataset_specs}
        if missing_d:
            raise ValueError(f"Unknown --datasets entries: {sorted(missing_d)}")

    if args.aggregate_only:
        out_root = project_dir / "results" / "multi_seed_agg"
        out_root.mkdir(parents=True, exist_ok=True)
        for ds in dataset_specs:
            for m in model_specs:
                paths = [
                    per_seed_csv_path(project_dir, ds, m.name, m.layer, s)
                    for s in seeds
                ]
                agg = aggregate_mean_std(
                    paths,
                    ddof=args.std_ddof,
                    model_label=f"{m.name}-l{m.layer}",
                )
                seed_tag = "-".join(str(s) for s in seeds)
                out_csv = (
                    out_root
                    / ds.key
                    / f"{m.name}-l{m.layer}-{ds.csv_middle}-seeds{seed_tag}_meanstd.csv"
                )
                out_csv.parent.mkdir(parents=True, exist_ok=True)
                agg.to_csv(out_csv, index=False)
                print(f"✓ Wrote {out_csv}")
        return

    for ds in dataset_specs:
        for m in model_specs:
            for seed in seeds:
                if not args.skip_generate:
                    for steer, use_t, t_val in METHODS:
                        k = manifold_k_for(m.name, ds.key, k_overrides)
                        vk = vk_path_for(project_dir, m.name, m.layer, k, ds.key)
                        if steer == "CRFDSteer":
                            if not args.skip_ensure_vk:
                                ensure_manifold_vk(
                                    project_dir,
                                    args.uv,
                                    m.name,
                                    m.layer,
                                    k,
                                    ds.key,
                                    args.dry_run,
                                )
                            vk_arg, k_arg = vk, k
                        else:
                            vk_arg, k_arg = None, None

                        t = resolve_steer_t(
                            steer, use_t, t_val, m.name, ds, t_overrides
                        )
                        cmd = build_generate_cmd(
                            args.uv,
                            ds,
                            m.name,
                            m.layer,
                            steer,
                            t,
                            seed,
                            vk_path=vk_arg,
                            manifold_dim=k_arg,
                        )
                        run_cmd(cmd, project_dir, args.dry_run)

                if not args.skip_eval:
                    run_cmd(
                        build_eval_cmd(
                            args.uv,
                            ds,
                            m.name,
                            m.layer,
                            seed,
                            reward_override=args.reward_model,
                        ),
                        project_dir,
                        args.dry_run,
                    )

    if not args.dry_run:
        out_root = project_dir / "results" / "multi_seed_agg"
        out_root.mkdir(parents=True, exist_ok=True)
        for ds in dataset_specs:
            for m in model_specs:
                paths = [
                    per_seed_csv_path(project_dir, ds, m.name, m.layer, s)
                    for s in seeds
                ]
                if not all(p.is_file() for p in paths):
                    print(
                        f"Warning: skip aggregate for {ds.key} {m.name}-l{m.layer}: "
                        f"not all seed CSVs exist yet",
                        flush=True,
                    )
                    continue
                agg = aggregate_mean_std(
                    paths,
                    ddof=args.std_ddof,
                    model_label=f"{m.name}-l{m.layer}",
                )
                seed_tag = "-".join(str(s) for s in seeds)
                out_csv = (
                    out_root
                    / ds.key
                    / f"{m.name}-l{m.layer}-{ds.csv_middle}-seeds{seed_tag}_meanstd.csv"
                )
                out_csv.parent.mkdir(parents=True, exist_ok=True)
                agg.to_csv(out_csv, index=False)
                print(f"✓ Wrote {out_csv}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
