"""
跨随机种子的 Manifold-k 敏感度曲线聚合：均值、样本标准差、bootstrap 均值 95% CI。

供 ``run_ode_manifold_k_sensitivity_*.py`` 复用；无外部 IO。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def parse_seeds(s: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_boot: int,
    ci: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """对观测值做 bootstrap 重采样均值的百分位 CI（双侧）。"""
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        m = float(x[0])
        return m, m
    alpha_lo = (1.0 - ci) / 2.0
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = float(np.mean(x[idx]))
    lo = float(np.quantile(boots, alpha_lo))
    hi = float(np.quantile(boots, 1.0 - alpha_lo))
    return lo, hi


def aggregate_metrics_by_k(
    seed_dfs: list[pd.DataFrame],
    metric_cols: list[str],
    *,
    k_col: str = "_k",
    n_boot: int,
    ci: float,
    bootstrap_seed: int,
) -> pd.DataFrame:
    """
    对每个在所有 seed 表格中均出现的 ``k``，汇总各指标的均值、样本标准差与 bootstrap CI。

    ``seed_dfs`` 须已含 ``k_col`` 且每个 ``k`` 至多一行。
    """
    if not seed_dfs:
        raise ValueError("seed_dfs is empty")
    ks_sets = [set(df[k_col].astype(int).unique()) for df in seed_dfs]
    common_ks = sorted(set.intersection(*ks_sets))
    if not common_ks:
        raise ValueError("No k intersection across seeds — check eval CSVs.")

    rng = np.random.default_rng(bootstrap_seed)
    rows: list[dict] = []

    for k in common_ks:
        row: dict = {k_col: k}
        vecs: dict[str, list[float]] = {c: [] for c in metric_cols}
        for df in seed_dfs:
            sub = df[df[k_col] == k]
            if sub.empty:
                raise RuntimeError(f"Missing k={k} in one of the seed frames.")
            if len(sub) > 1:
                raise RuntimeError(f"Duplicate rows for k={k} in a seed frame.")
            r = sub.iloc[0]
            for c in metric_cols:
                vecs[c].append(float(r[c]))

        for c in metric_cols:
            arr = np.array(vecs[c], dtype=float)
            n_s = arr.size
            ddof = 0 if n_s < 2 else 1
            row[f"{c}_mean"] = float(np.mean(arr))
            row[f"{c}_std"] = float(np.std(arr, ddof=ddof))
            lo, hi = bootstrap_mean_ci(arr, n_boot=n_boot, ci=ci, rng=rng)
            row[f"{c}_ci_lo"] = lo
            row[f"{c}_ci_hi"] = hi
        rows.append(row)

    return pd.DataFrame(rows).sort_values(k_col).reset_index(drop=True)
