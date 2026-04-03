"""Strip ODE-Manifold CLI flags before Hydra, and merge into steer kwargs."""

from __future__ import annotations

import argparse
import sys
from typing import Any

import torch


def strip_manifold_flags_from_argv(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse and remove ODE-Manifold CLI flags from argv so Hydra can parse the remainder.

    - ``--use_manifold``: load ``V_k`` and enable tangent-space projection.
    - ``--use_manifold_retraction``: enable norm-preserving retraction; with ``--use_manifold`` = full
      manifold; **without** ``--use_manifold`` = Euclidean ODE + retraction only (ablation).
    - ``--vk_path``, ``--manifold_dim``
    """
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--use_manifold", action="store_true")
    parser.add_argument("--use_manifold_retraction", action="store_true")
    parser.add_argument("--manifold_dim", type=int, default=128)
    parser.add_argument("--vk_path", type=str, default=None)
    args, remaining = parser.parse_known_args(argv)
    sys.argv = [sys.argv[0]] + remaining
    return args


def merge_manifold_cli_into_steer_kwargs(
    steer_kwargs: dict[str, Any],
    cli_args: argparse.Namespace,
) -> dict[str, Any]:
    """
    If ``--use_manifold``, load ``V_k`` and set flags for ablations:

    - ``--use_manifold`` only → projection only, ``use_manifold_retraction=False``.
    - ``--use_manifold --use_manifold_retraction`` → projection + retraction (recommended).

    If **only** ``--use_manifold_retraction`` (no ``--use_manifold``): set
    ``use_manifold_retraction=True`` for Euclidean ODE + norm retraction (no ``V_k``).
    """
    out = dict(steer_kwargs)
    if getattr(cli_args, "use_manifold", False):
        out["use_manifold"] = True
        out["use_manifold_retraction"] = bool(getattr(cli_args, "use_manifold_retraction", False))
        vk_path = cli_args.vk_path or out.get("vk_path")
        if not vk_path:
            raise ValueError(
                "--use_manifold requires --vk_path=... or steer.kwargs.vk_path in Hydra config"
            )
        Vk = torch.load(vk_path, map_location="cpu", weights_only=False)
        if isinstance(Vk, dict) and "V_k" in Vk:
            Vk = Vk["V_k"]
        md = int(cli_args.manifold_dim)
        from odesteer.steer._manifold import truncate_vk

        out["V_k"] = truncate_vk(Vk, md)
        out["vk_path"] = vk_path
        out["manifold_dim"] = md
        return out
    if getattr(cli_args, "use_manifold_retraction", False):
        out["use_manifold_retraction"] = True
    return out


def steer_name_with_manifold_suffix(base_steer_name: str, steer_model_kwargs: dict[str, Any]) -> str:
    """
    Append a distinct suffix so outputs / eval rows do not collide:

    - Manifold (``use_manifold`` or ``V_k``): ``-Manifold-k{k}`` (with retraction) or
      ``-Manifold-k{k}-proj`` (projection only).
    - **Euclidean + norm retraction only** (``use_manifold_retraction`` but no ``V_k`` / no
      ``use_manifold``): ``-NormRetr`` — same vector field as vanilla ODESteer, then scale back
      to input L2 norm (ablation for projection).
    """
    use_m = bool(steer_model_kwargs.get("use_manifold"))
    has_vk = steer_model_kwargs.get("V_k") is not None
    retr = bool(steer_model_kwargs.get("use_manifold_retraction"))
    k = int(steer_model_kwargs.get("manifold_dim", 128))

    if use_m or has_vk:
        token = f"Manifold-k{k}" if retr else f"Manifold-k{k}-proj"
        if base_steer_name.endswith(token):
            return base_steer_name
        return f"{base_steer_name}-{token}"

    if retr:
        token = "NormRetr"
        if base_steer_name.endswith(token):
            return base_steer_name
        return f"{base_steer_name}-{token}"

    return base_steer_name
