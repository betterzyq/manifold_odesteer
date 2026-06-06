#!/usr/bin/env python3
"""
Llama3.1-8B-Base 在三个 alignment 任务上的中间层激活几何可视化与诊断。

- 点云：正/负类激活在 PCA / t-SNE / UMAP（可选）下的分布
- 流形：V_k 系数空间、离流形残差、PCA 方差谱
- Steering 轨迹：CAA / RepE / ITI / ODESteer（无流形）/ CRFDSteer（+Manifold）

用法（仓库根目录）：
  uv run python scripts/visualization/plot_llama_activation_geometry.py
  uv run python scripts/visualization/plot_llama_activation_geometry.py --dims 3 --embed pca
  uv run python scripts/visualization/plot_llama_activation_geometry.py --tasks truthfulqa --max-fit 1500
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection
import torch
from matplotlib.lines import Line2D
from omegaconf import OmegaConf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch import Tensor
from torchdiffeq import odeint

from odesteer.steer import CAA, CRFDSteer, ITI, RepE
from odesteer.steer._manifold import load_vk_from_path
from odesteer.steer._ode_steer import BaseCRFDSteer
from odesteer.utils import get_project_dir
from odesteer.utils.data import (
    load_jigsaw_activations,
    load_tqa_gen_data,
    load_ultrafeedback_data,
)

try:
    import umap

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

MODEL = "Llama3.1-8B-Base"
LAYER = 13
STEER_T = 5.0  # Llama 三任务默认（与 run_multi_seed_grid BASE_CRFD_T 一致）

MANIFOLD_K: dict[str, int] = {
    "truthfulqa": 256,
    "toxicity": 128,
    "ultrafeedback": 32,
}

TASK_LABELS: dict[str, str] = {
    "truthfulqa": "TruthfulQA",
    "toxicity": "Real Toxicity Prompts",
    "ultrafeedback": "UltraFeedback",
}

METHOD_STYLES: dict[str, dict] = {
    "CAA": {"color": "#1f77b4", "ls": "-", "lw": 1.8},
    "RepE": {"color": "#2ca02c", "ls": "-", "lw": 1.5},
    "ITI": {"color": "#ff7f0e", "ls": "-", "lw": 1.5},
    "ODESteer": {"color": "#9467bd", "ls": "--", "lw": 1.8},
    "CRFDSteer": {"color": "#d62728", "ls": "-", "lw": 2.2},
}

POS_COLOR = "#4C72B0"
NEG_COLOR = "#C44E52"


@dataclass
class TaskData:
    key: str
    pos: Tensor
    neg: Tensor
    vk_path: Path
    manifold_k: int


def load_task_data(task: str, model: str, layer: int) -> TaskData:
    root = get_project_dir() / "data"
    k = MANIFOLD_K[task]
    if task == "truthfulqa":
        pos, neg = load_tqa_gen_data(model, layer, split_idx=0)
        act_dir = root / "truthfulqa" / "activations" / model
    elif task == "toxicity":
        pos, neg = load_jigsaw_activations(model, layer)
        act_dir = root / "toxicity" / "activations" / model
    elif task == "ultrafeedback":
        pos, neg = load_ultrafeedback_data(model, layer, split="train")
        act_dir = root / "ultrafeedback" / "activations" / model
    else:
        raise ValueError(task)
    vk = act_dir / f"manifold_Vk_layer{layer}_k{k}.pt"
    if not vk.is_file():
        raise FileNotFoundError(vk)
    return TaskData(task, pos.float(), neg.float(), vk, k)


def subsample_pair(
    pos: Tensor,
    neg: Tensor,
    n_per_class: int,
    seed: int,
    max_fit: int | None,
) -> tuple[Tensor, Tensor]:
    rng = np.random.default_rng(seed)
    n_pos = min(n_per_class, len(pos))
    n_neg = min(n_per_class, len(neg))
    ip = torch.from_numpy(rng.choice(len(pos), n_pos, replace=False))
    ineg = torch.from_numpy(rng.choice(len(neg), n_neg, replace=False))
    pos_s, neg_s = pos[ip], neg[ineg]
    if max_fit is not None:
        mf = max_fit // 2
        pos_s = pos_s[: min(len(pos_s), mf)]
        neg_s = neg_s[: min(len(neg_s), mf)]
    return pos_s, neg_s


def crfd_kwargs() -> dict:
    cfg = OmegaConf.load(get_project_dir() / "confs" / "steer" / "CRFDSteer.yaml")
    return dict(OmegaConf.to_container(cfg.kwargs, resolve=True))


def fit_all_steers(
    pos: Tensor,
    neg: Tensor,
    vk_path: Path,
    manifold_k: int,
    traj_device: torch.device,
) -> dict[str, object]:
    """在 CPU 上 fit（CRFD 分类器走 sklearn）；轨迹积分在 traj_device。"""
    pos_cpu, neg_cpu = pos.cpu(), neg.cpu()
    out: dict[str, object] = {}
    out["CAA"] = CAA().fit(pos_cpu, neg_cpu)
    out["RepE"] = RepE().fit(pos_cpu, neg_cpu)
    out["ITI"] = ITI().fit(pos_cpu, neg_cpu)

    base_kw = crfd_kwargs()
    out["ODESteer"] = CRFDSteer(**{**base_kw, "use_manifold": False}).fit(pos_cpu, neg_cpu)
    mf_kw = {
        **base_kw,
        "use_manifold": True,
        "use_manifold_retraction": True,
        "vk_path": str(vk_path),
        "manifold_dim": manifold_k,
    }
    out["CRFDSteer"] = CRFDSteer(**mf_kw).fit(pos_cpu, neg_cpu)
    for s in out.values():
        if hasattr(s, "V_k") and s.V_k is not None:
            s.V_k = s.V_k.to(traj_device)
    return out


@torch.no_grad()
def trajectory_points(
    steer: object,
    x0: Tensor,
    T: float,
    n_steps: int,
) -> Tensor:
    """返回 [n_steps+1, d] 轨迹（含起点）。"""
    x0 = x0.unsqueeze(0) if x0.dim() == 1 else x0
    device = x0.device
    if isinstance(steer, BaseCRFDSteer):
        ts = torch.linspace(0.0, T, n_steps + 1, device=device)
        traj = odeint(
            lambda t, state: steer.vector_field(state),
            x0,
            ts,
            method=steer.solver,
            options={"step_size": T / steer.steps},
        )
        return traj[:, 0, :].cpu()
    # VecSteer：直线路径
    vf = steer.vector_field(x0)
    alphas = torch.linspace(0.0, T, n_steps + 1, device=device).view(-1, 1, 1)
    path = x0.unsqueeze(0) + alphas * vf.unsqueeze(0)
    return path[:, 0, :].cpu()


def collect_trajectories(
    steers: dict[str, object],
    anchors: Tensor,
    T: float,
    n_steps: int,
) -> dict[str, np.ndarray]:
    """每个方法: [n_anchors, n_steps+1, d]"""
    device = anchors.device
    out = {}
    for name, steer in steers.items():
        paths = []
        for i in range(len(anchors)):
            paths.append(
                trajectory_points(steer, anchors[i].to(device), T, n_steps).numpy()
            )
        out[name] = np.stack(paths, axis=0)
    return out


def off_manifold_norm(X: np.ndarray, Vk: np.ndarray) -> np.ndarray:
    """||x - P_{V_k} x||_2"""
    Vk = Vk.astype(np.float64)
    coeff = X @ Vk
    proj = coeff @ Vk.T
    return np.linalg.norm(X - proj, axis=1)


def embed_nd(
    X: np.ndarray,
    method: str,
    n_components: int,
    seed: int,
    perplexity: float,
) -> np.ndarray:
    if n_components not in (2, 3):
        raise ValueError("n_components must be 2 or 3")
    if method == "pca":
        return PCA(n_components=n_components, random_state=seed).fit_transform(X)
    if method == "tsne":
        perp = min(perplexity, max(5.0, (len(X) - 1) / 3))
        return TSNE(
            n_components=n_components,
            perplexity=perp,
            init="pca",
            learning_rate="auto",
            random_state=seed,
        ).fit_transform(X)
    if method == "umap":
        if not HAS_UMAP:
            raise RuntimeError("umap-learn 未安装")
        return umap.UMAP(
            n_components=n_components,
            n_neighbors=min(30, len(X) - 1),
            min_dist=0.1,
            metric="cosine",
            random_state=seed,
        ).fit_transform(X)
    raise ValueError(method)


def _suffix_dim(dim: int) -> str:
    return "" if dim == 2 else "_3d"


def _draw_steering_on_axes(
    ax,
    emb: np.ndarray,
    tags: np.ndarray,
    trajectories_sub: dict[str, np.ndarray],
    *,
    dim: int,
) -> list:
    pos_m = tags == "pos"
    neg_m = tags == "neg"
    if dim == 2:
        ax.scatter(
            emb[pos_m, 0], emb[pos_m, 1],
            c=POS_COLOR, s=8, alpha=0.35, rasterized=True,
        )
        ax.scatter(
            emb[neg_m, 0], emb[neg_m, 1],
            c=NEG_COLOR, s=8, alpha=0.35, rasterized=True,
        )
    else:
        ax.scatter(
            emb[pos_m, 0], emb[pos_m, 1], emb[pos_m, 2],
            c=POS_COLOR, s=6, alpha=0.3, depthshade=True,
        )
        ax.scatter(
            emb[neg_m, 0], emb[neg_m, 1], emb[neg_m, 2],
            c=NEG_COLOR, s=6, alpha=0.3, depthshade=True,
        )

    offset = int(pos_m.sum() + neg_m.sum())
    cursor = offset
    n_anc = next(iter(trajectories_sub.values())).shape[0]
    n_t = next(iter(trajectories_sub.values())).shape[1]

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=POS_COLOR, markersize=6, label="Positive"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=NEG_COLOR, markersize=6, label="Negative"),
    ]
    for name, _traj in trajectories_sub.items():
        style = METHOD_STYLES[name]
        for a in range(n_anc):
            idx = np.arange(cursor + a * n_t, cursor + (a + 1) * n_t)
            if dim == 2:
                ax.plot(
                    emb[idx, 0], emb[idx, 1],
                    color=style["color"], ls=style["ls"], lw=style["lw"], alpha=0.85,
                )
                ax.scatter(emb[idx[0], 0], emb[idx[0], 1], c=style["color"], s=18, marker="o", zorder=5)
                ax.scatter(emb[idx[-1], 0], emb[idx[-1], 1], c=style["color"], s=28, marker="*", zorder=5)
            else:
                ax.plot(
                    emb[idx, 0], emb[idx, 1], emb[idx, 2],
                    color=style["color"], ls=style["ls"], lw=style["lw"], alpha=0.9,
                )
                ax.scatter(
                    emb[idx[0], 0], emb[idx[0], 1], emb[idx[0], 2],
                    c=style["color"], s=20, marker="o", depthshade=False,
                )
                ax.scatter(
                    emb[idx[-1], 0], emb[idx[-1], 1], emb[idx[-1], 2],
                    c=style["color"], s=40, marker="*", depthshade=False,
                )
        cursor += n_anc * n_t
        legend_handles.append(
            Line2D([0], [0], color=style["color"], ls=style["ls"], lw=style["lw"], label=name)
        )
    return legend_handles


def subsample_trajectories(
    trajectories: dict[str, np.ndarray],
    n_traj_pts: int,
) -> dict[str, np.ndarray]:
    out = {}
    for name, traj in trajectories.items():
        n_anc, n_t, d = traj.shape
        step_idx = np.linspace(0, n_t - 1, n_traj_pts, dtype=int)
        out[name] = traj[:, step_idx, :]
    return out


def build_embedding_matrix(
    pos: np.ndarray,
    neg: np.ndarray,
    trajectories_sub: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """拼接点云 + 轨迹子采样，供联合降维。"""
    parts = [pos, neg]
    tags = ["pos"] * len(pos) + ["neg"] * len(neg)
    for name, traj in trajectories_sub.items():
        n_anc, n_t, d = traj.shape
        flat = traj.reshape(-1, d)
        parts.append(flat)
        tags.extend([f"traj:{name}"] * len(flat))
    return np.vstack(parts), np.array(tags, dtype=object)


def plot_embedding_panel(
    emb: np.ndarray,
    tags: np.ndarray,
    trajectories_sub: dict[str, np.ndarray],
    method: str,
    task_label: str,
    out_path: Path,
    *,
    dim: int = 2,
    elev: float = 24.0,
    azim: float = -58.0,
) -> None:
    dim_tag = "3D" if dim == 3 else method.upper()
    if dim == 2:
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        legend_handles = _draw_steering_on_axes(ax, emb, tags, trajectories_sub, dim=2)
        ax.set_title(f"{task_label} — layer {LAYER} ({method.upper()})")
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")
        ax.legend(handles=legend_handles, loc="best", fontsize=7, framealpha=0.9)
        ax.grid(True, alpha=0.2)
    else:
        fig = plt.figure(figsize=(9.0, 8.0))
        ax = fig.add_subplot(111, projection="3d")
        legend_handles = _draw_steering_on_axes(ax, emb, tags, trajectories_sub, dim=3)
        ax.set_title(f"{task_label} — layer {LAYER} ({method.upper()} 3D)")
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")
        ax.set_zlabel("dim 3")
        ax.view_init(elev=elev, azim=azim)
        ax.legend(handles=legend_handles, loc="upper left", fontsize=7, framealpha=0.9)
        # 弱化 panes 网格，突出点云
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_activation_cloud(
    emb: np.ndarray,
    n_pos: int,
    emb_name: str,
    task_label: str,
    out_path: Path,
    *,
    dim: int = 2,
    elev: float = 24.0,
    azim: float = -58.0,
) -> None:
    if dim == 2:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(emb[:n_pos, 0], emb[:n_pos, 1], c=POS_COLOR, s=10, alpha=0.4, label="Positive")
        ax.scatter(emb[n_pos:, 0], emb[n_pos:, 1], c=NEG_COLOR, s=10, alpha=0.4, label="Negative")
        ax.grid(True, alpha=0.2)
    else:
        fig = plt.figure(figsize=(8.5, 7.5))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            emb[:n_pos, 0], emb[:n_pos, 1], emb[:n_pos, 2],
            c=POS_COLOR, s=8, alpha=0.35, depthshade=True, label="Positive",
        )
        ax.scatter(
            emb[n_pos:, 0], emb[n_pos:, 1], emb[n_pos:, 2],
            c=NEG_COLOR, s=8, alpha=0.35, depthshade=True, label="Negative",
        )
        ax.set_zlabel("dim 3")
        ax.view_init(elev=elev, azim=azim)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(True, alpha=0.25)
    suffix = "3D" if dim == 3 else emb_name.upper()
    ax.set_title(f"{task_label} — activation cloud ({suffix})")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pca_scree(ev_ratio: np.ndarray, task_label: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    k = min(50, len(ev_ratio))
    xs = np.arange(1, k + 1)
    ax.bar(xs, ev_ratio[:k], color="#4C72B0", alpha=0.85)
    ax.plot(xs, np.cumsum(ev_ratio[:k]), "o-", color="#C44E52", ms=3, label="cumulative")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title(f"{task_label} — PCA spectrum (d={len(ev_ratio)})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_off_manifold(
    pos_off: np.ndarray,
    neg_off: np.ndarray,
    task_label: str,
    k: int,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bins = np.linspace(
        0, max(pos_off.max(), neg_off.max()) * 1.05, 40
    )
    ax.hist(pos_off, bins=bins, alpha=0.55, color=POS_COLOR, label="Positive", density=True)
    ax.hist(neg_off, bins=bins, alpha=0.55, color=NEG_COLOR, label="Negative", density=True)
    ax.set_xlabel(r"$\|x - P_{V_k}x\|_2$")
    ax.set_ylabel("density")
    ax.set_title(f"{task_label} — off-manifold residual (k={k})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_manifold_coefficients(
    pos: np.ndarray,
    neg: np.ndarray,
    Vk: np.ndarray,
    task_label: str,
    out_path: Path,
    *,
    dim: int = 2,
) -> None:
    coeff_pos = pos @ Vk
    coeff_neg = neg @ Vk
    combined = np.vstack([coeff_pos, coeff_neg])
    n_p = len(coeff_pos)
    n_comp = 3 if dim == 3 else 2
    emb = PCA(n_components=n_comp, random_state=0).fit_transform(combined)
    if dim == 2:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(emb[:n_p, 0], emb[:n_p, 1], c=POS_COLOR, s=10, alpha=0.4, label="Positive")
        ax.scatter(emb[n_p:, 0], emb[n_p:, 1], c=NEG_COLOR, s=10, alpha=0.4, label="Negative")
        ax.set_xlabel("PC1 (coeff)")
        ax.set_ylabel("PC2 (coeff)")
        ax.grid(True, alpha=0.2)
    else:
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            emb[:n_p, 0], emb[:n_p, 1], emb[:n_p, 2],
            c=POS_COLOR, s=10, alpha=0.4, depthshade=True, label="Positive",
        )
        ax.scatter(
            emb[n_p:, 0], emb[n_p:, 1], emb[n_p:, 2],
            c=NEG_COLOR, s=10, alpha=0.4, depthshade=True, label="Negative",
        )
        ax.set_xlabel("PC1 (coeff)")
        ax.set_ylabel("PC2 (coeff)")
        ax.set_zlabel("PC3 (coeff)")
        ax.view_init(elev=22, azim=-55)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(True, alpha=0.25)
    title_suffix = "3D" if dim == 3 else "2D"
    ax.set_title(f"{task_label} — PCA in $V_k$ coeffs (k={Vk.shape[1]}, {title_suffix})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def process_task(
    td: TaskData,
    out_dir: Path,
    *,
    n_cloud: int,
    n_anchors: int,
    n_steps: int,
    max_fit: int | None,
    seed: int,
    embed_methods: list[str],
    dims: list[int],
    alt_views: bool,
    device: torch.device,
) -> dict:
    label = TASK_LABELS[td.key]
    task_out = out_dir / td.key
    task_out.mkdir(parents=True, exist_ok=True)

    pos_s, neg_s = subsample_pair(td.pos, td.neg, n_cloud, seed, max_fit)
    pos_np = pos_s.numpy()
    neg_np = neg_s.numpy()

    # 诊断：全量点云 PCA 谱（最多 4000 点）
    diag_n = min(2000, len(td.pos), len(td.neg))
    rng = np.random.default_rng(seed)
    ip = rng.choice(len(td.pos), min(diag_n, len(td.pos)), replace=False)
    ineg = rng.choice(len(td.neg), min(diag_n, len(td.neg)), replace=False)
    X_diag = torch.cat([td.pos[ip], td.neg[ineg]]).numpy()
    pca_full = PCA().fit(X_diag)
    plot_pca_scree(
        pca_full.explained_variance_ratio_,
        label,
        task_out / "pca_scree.png",
    )

    Vk = load_vk_from_path(td.vk_path, td.manifold_k).numpy()
    pos_off = off_manifold_norm(pos_np, Vk)
    neg_off = off_manifold_norm(neg_np, Vk)
    plot_off_manifold(pos_off, neg_off, label, td.manifold_k, task_out / "off_manifold_hist.png")
    plot_manifold_coefficients(pos_np, neg_np, Vk, label, task_out / "manifold_coeff_pca.png", dim=2)
    if 3 in dims:
        plot_manifold_coefficients(
            pos_np, neg_np, Vk, label, task_out / "manifold_coeff_pca_3d.png", dim=3
        )

    # Steering：从负类锚点出发
    rng_t = np.random.default_rng(seed + 1)
    anc_idx = rng_t.choice(len(neg_s), min(n_anchors, len(neg_s)), replace=False)
    anchors = neg_s[anc_idx]

    pos_fit, neg_fit = subsample_pair(td.pos, td.neg, n_cloud, seed + 2, max_fit)
    steers = fit_all_steers(pos_fit, neg_fit, td.vk_path, td.manifold_k, device)
    # 轨迹积分放 GPU（若可用）
    trajectories = collect_trajectories(steers, anchors, STEER_T, n_steps)

    stats = {
        "task": td.key,
        "n_pos_cloud": len(pos_np),
        "n_neg_cloud": len(neg_np),
        "manifold_k": td.manifold_k,
        "steer_T": STEER_T,
        "pca_var_top5": pca_full.explained_variance_ratio_[:5].tolist(),
        "pca_var_top32_cum": float(np.cumsum(pca_full.explained_variance_ratio_)[31]),
        "off_manifold_mean_pos": float(pos_off.mean()),
        "off_manifold_mean_neg": float(neg_off.mean()),
        "center_dist_pos_neg": float(np.linalg.norm(pos_np.mean(0) - neg_np.mean(0))),
    }

    n_traj_pts = min(12, n_steps + 1)
    traj_sub = subsample_trajectories(trajectories, n_traj_pts)
    views = [(24.0, -58.0), (18.0, 128.0)] if alt_views else [(24.0, -58.0)]

    for emb_name in embed_methods:
        X_all, tags = build_embedding_matrix(pos_np, neg_np, traj_sub)
        for dim in dims:
            if emb_name == "tsne" and len(X_all) > 3000 and dim == 3:
                continue
            emb = embed_nd(X_all, emb_name, dim, seed, perplexity=40.0)
            sd = _suffix_dim(dim)
            for vi, (elev, azim) in enumerate(views):
                vtag = f"_view{vi}" if (dim == 3 and alt_views and len(views) > 1) else ""
                plot_embedding_panel(
                    emb,
                    tags,
                    traj_sub,
                    emb_name,
                    label,
                    task_out / f"steering_trajectories_{emb_name}{sd}{vtag}.png",
                    dim=dim,
                    elev=elev,
                    azim=azim,
                )

    cloud = np.vstack([pos_np, neg_np])
    for emb_name in embed_methods:
        if emb_name == "tsne" and len(cloud) > 3000:
            continue
        for dim in dims:
            emb_cloud = embed_nd(cloud, emb_name, dim, seed, perplexity=40.0)
            sd = _suffix_dim(dim)
            for vi, (elev, azim) in enumerate(views if dim == 3 else [(24.0, -58.0)]):
                vtag = f"_view{vi}" if (dim == 3 and alt_views and len(views) > 1) else ""
                plot_activation_cloud(
                    emb_cloud,
                    len(pos_np),
                    emb_name,
                    label,
                    task_out / f"activation_cloud_{emb_name}{sd}{vtag}.png",
                    dim=dim,
                    elev=elev,
                    azim=azim,
                )

    with open(task_out / "diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return stats


def plot_summary_row(
    out_dir: Path,
    tasks: list[str],
    embed_method: str = "pca",
    *,
    dim: int = 2,
) -> None:
    """三任务 steering 轨迹横排总览。"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sd = _suffix_dim(dim)
    for ax, task in zip(axes, tasks):
        img_path = out_dir / task / f"steering_trajectories_{embed_method}{sd}.png"
        if not img_path.is_file():
            ax.set_visible(False)
            continue
        img = plt.imread(img_path)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(TASK_LABELS[task], fontsize=10)
    title_dim = "3D" if dim == 3 else embed_method.upper()
    fig.suptitle(
        f"Llama3.1-8B layer {LAYER} — steering trajectories ({title_dim})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_dir / f"summary_steering_{embed_method}{sd}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=list(TASK_LABELS.keys()))
    p.add_argument("--n-cloud", type=int, default=600, help="每类点云可视化样本数")
    p.add_argument("--n-anchors", type=int, default=10, help="轨迹起点数（从负类抽取）")
    p.add_argument("--n-steps", type=int, default=10, help="轨迹离散步数（与 CRFD steps 对齐）")
    p.add_argument("--max-fit", type=int, default=2000, help="拟合 steer 每类最多样本；None=全量")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--embed",
        nargs="+",
        default=["pca", "tsne"] + (["umap"] if HAS_UMAP else []),
        choices=["pca", "tsne", "umap"],
    )
    p.add_argument(
        "--dims",
        default="2,3",
        help="输出维度，逗号分隔，如 2 / 3 / 2,3",
    )
    p.add_argument(
        "--no-alt-views",
        action="store_true",
        help="3D 图不保存第二视角（默认保存 view0+view1）",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir or (
        get_project_dir() / "results" / "figures" / "activation_geometry" / MODEL
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dims = sorted({int(x.strip()) for x in args.dims.split(",") if x.strip()})
    if not dims or any(d not in (2, 3) for d in dims):
        raise ValueError("--dims 仅支持 2 与/或 3")
    alt_views = not args.no_alt_views

    all_stats = []
    for task in args.tasks:
        print(f"\n→ Task: {task} (dims={dims})")
        td = load_task_data(task, MODEL, LAYER)
        stats = process_task(
            td,
            out_dir,
            n_cloud=args.n_cloud,
            n_anchors=args.n_anchors,
            n_steps=args.n_steps,
            max_fit=args.max_fit,
            seed=args.seed,
            embed_methods=args.embed,
            dims=dims,
            alt_views=alt_views,
            device=device,
        )
        all_stats.append(stats)
        print(f"  PCA top-5 var: {stats['pca_var_top5']}")
        print(f"  off-manifold pos/neg: {stats['off_manifold_mean_pos']:.4f} / {stats['off_manifold_mean_neg']:.4f}")

    with open(out_dir / "all_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)

    if len(args.tasks) >= 3:
        for dim in dims:
            plot_summary_row(out_dir, args.tasks[:3], "pca", dim=dim)
            if "umap" in args.embed and HAS_UMAP:
                plot_summary_row(out_dir, args.tasks[:3], "umap", dim=dim)

    print(f"\n→ Figures saved under: {out_dir}")


if __name__ == "__main__":
    main()
