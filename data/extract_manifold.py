"""
Offline SVD on layer activations to build an orthogonal basis V_k [d, k] for ODE-Manifold.

Collects all *activations_layer{L}.pt tensors under data/<task>/activations/<model>/,
concatenates to X [N, d], runs torch.linalg.svd, saves V_k = Vh[:k].T.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from odesteer.utils import get_project_dir


def collect_activation_matrix(activations_dir: Path, layer_idx: int) -> torch.Tensor:
    pattern = f"*activations_layer{layer_idx}.pt"
    files = sorted(activations_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matching {pattern!r} under {activations_dir}. "
            "Run the corresponding data/*/extract_activations.py first."
        )
    parts: list[torch.Tensor] = []
    for f in files:
        t = torch.load(f, map_location="cpu", weights_only=False)
        if t.dim() != 2:
            raise ValueError(f"{f}: expected 2D [N, d], got {tuple(t.shape)}")
        parts.append(t.float())
    return torch.cat(parts, dim=0)


def compute_vk_from_X(X: torch.Tensor, k: int) -> torch.Tensor:
    """X: [N, d] -> V_k: [d, k] with orthonormal columns (first k right singular vectors)."""
    if k < 1:
        raise ValueError("k must be >= 1")
    # full_matrices=False: U [N, r], S [r], Vh [r, d], r = min(N, d)
    _u, _s, vh = torch.linalg.svd(X.float(), full_matrices=False)
    r = vh.shape[0]
    k_eff = min(k, r)
    if k_eff < k:
        print(f"Warning: requested k={k} but only r={r} singular values; using k_eff={k_eff}")
    return vh[:k_eff].T.contiguous()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract manifold basis V_k via SVD on activations")
    parser.add_argument("-m", "--model", type=str, default="Llama3.1-8B-Base")
    parser.add_argument("-l", "--layer_idx", type=int, required=True, help="Layer index (same as steering layer)")
    parser.add_argument(
        "-k",
        "--manifold_dim",
        type=int,
        default=128,
        help="Number of right singular vectors to keep (columns of V_k)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="truthfulqa",
        choices=("truthfulqa", "ultrafeedback", "toxicity"),
        help="Which data/*/activations/<model>/ directory to read",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output .pt path (default: <activations_dir>/manifold_Vk_layer{L}_k{K}.pt)",
    )
    parser.add_argument(
        "--activations_dir",
        type=str,
        default=None,
        help="Override directory containing *activations_layer{L}.pt",
    )
    args = parser.parse_args()

    project_dir = get_project_dir()
    if args.activations_dir:
        activations_dir = Path(args.activations_dir).resolve()
    else:
        activations_dir = project_dir / "data" / args.task / "activations" / args.model
    if not activations_dir.is_dir():
        raise FileNotFoundError(f"activations_dir does not exist: {activations_dir}")

    print(f"→ Collecting activations from {activations_dir} (layer {args.layer_idx}) ...")
    X = collect_activation_matrix(activations_dir, args.layer_idx)
    print(f"  Stacked X shape: {tuple(X.shape)}")

    print(f"→ SVD and building V_k with k={args.manifold_dim} ...")
    Vk = compute_vk_from_X(X, args.manifold_dim)
    print(f"  V_k shape: {tuple(Vk.shape)} (d={Vk.shape[0]}, k={Vk.shape[1]})")

    out_path = (
        Path(args.output).resolve()
        if args.output
        else activations_dir / f"manifold_Vk_layer{args.layer_idx}_k{args.manifold_dim}.pt"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(Vk, out_path)
    print(f"✓ Saved V_k to {out_path}")


if __name__ == "__main__":
    main()
