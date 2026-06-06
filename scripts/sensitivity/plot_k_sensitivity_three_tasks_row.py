#!/usr/bin/env python3
"""
从各任务 ``*_bootci_aggregate.csv`` 直接绘制 k 敏感度曲线，并拼成一行主图。

顺序：Ultrafeedback | TruthfulQA | Real Toxicity Prompts
仅上方显示数据集名称；坐标轴与图例字号大于标题。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

from odesteer.utils import get_project_dir

# 版面与字号（矢量出图，不缩放位图）
FIGSIZE = (10.5, 2.85)  # 英寸，一行三图
DPI = 300
FONT_TITLE = 10         # 数据集名
FONT_LABEL = 8         # 坐标轴名（略小于上一版，仍可读）
FONT_TICK = 9           # 刻度 / x 轴下方标签
FONT_LEGEND = 6        # 图例
MARKER_SIZE = 4
LINE_WIDTH = 1.6


def plot_k_panel(
    ax: plt.Axes,
    agg: pd.DataFrame,
    *,
    mean_col: str,
    lo_col: str,
    hi_col: str,
    ylabel: str,
    color: str,
    mean_label: str,
    ci_label: str = "bootstrap 95% CI",
) -> None:
    agg = agg.sort_values("_k").reset_index(drop=True)
    x = agg["_k"].astype(int).tolist()
    pos = list(range(len(x)))

    ax.plot(
        pos,
        agg[mean_col].tolist(),
        color=color,
        linestyle="-",
        marker="o",
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        label=mean_label,
        zorder=3,
    )
    ax.fill_between(
        pos,
        agg[lo_col].tolist(),
        agg[hi_col].tolist(),
        color=color,
        alpha=0.22,
        linewidth=0,
        label=ci_label,
        zorder=2,
    )
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
    ax.set_xlabel("Manifold subspace dimension $k$", fontsize=FONT_LABEL)
    ax.set_xticks(pos)
    ax.set_xticklabels([str(v) for v in x], rotation=40, ha="right", fontsize=FONT_TICK)
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(
        loc="best",
        fontsize=FONT_LEGEND,
        frameon=True,
        borderpad=0.35,
        handlelength=1.6,
        handletextpad=0.5,
        markerscale=0.85,
    )
    ax.grid(False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot 3-task k-sensitivity row from aggregate CSVs")
    parser.add_argument(
        "--output",
        type=str,
        default="results/figures/main_figure_k_sensitivity_three_tasks_row.png",
    )
    args = parser.parse_args()

    project = get_project_dir()
    uf = pd.read_csv(
        project
        / "results/ultrafeedback/sensitivity/manifold_k_crfdsteer_T5.0_l13_seeds42-123-456_bootci_aggregate.csv"
    )
    tqa = pd.read_csv(
        project
        / "results/truthfulqa/sensitivity/manifold_k_crfdsteer_T5.0_l13_seeds42-123-456_bootci_aggregate.csv"
    )
    tox = pd.read_csv(
        project
        / "results/toxicity/sensitivity/manifold_k_crfdsteer_T5.0_l13_seeds42-123-456_bootci_aggregate.csv"
    )

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "axes.linewidth": 0.8,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_facecolor("white")

    plot_k_panel(
        axes[0],
        uf,
        mean_col="RM Win-Rate vs NoSteer_mean",
        lo_col="RM Win-Rate vs NoSteer_ci_lo",
        hi_col="RM Win-Rate vs NoSteer_ci_hi",
        ylabel="RM Win-Rate vs NoSteer",
        color="#ff7f0e",
        mean_label="Win-Rate (mean)",
    )
    axes[0].set_title("Ultrafeedback", fontsize=FONT_TITLE, pad=4)

    plot_k_panel(
        axes[1],
        tqa,
        mean_col="True * Info_mean",
        lo_col="True * Info_ci_lo",
        hi_col="True * Info_ci_hi",
        ylabel="True × Info",
        color="#1f77b4",
        mean_label="True × Info (mean)",
    )
    axes[1].set_title("TruthfulQA", fontsize=FONT_TITLE, pad=4)

    plot_k_panel(
        axes[2],
        tox,
        mean_col="Toxicity_mean",
        lo_col="Toxicity_ci_lo",
        hi_col="Toxicity_ci_hi",
        ylabel="Toxicity",
        color="#d62728",
        mean_label="Toxicity (mean)",
    )
    axes[2].set_title("Real Toxicity Prompts", fontsize=FONT_TITLE, pad=4)

    plt.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.22, wspace=0.32)

    out = project / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white", pad_inches=0.04)
    plt.close(fig)
    print(f"✓ Saved {out} ({DPI} dpi)")


if __name__ == "__main__":
    main()
