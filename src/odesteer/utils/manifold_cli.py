"""Strip ODE-Manifold CLI flags before Hydra, and merge into steer kwargs."""

from __future__ import annotations

import argparse
import sys
from typing import Any

import torch


def strip_manifold_flags_from_argv(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse and remove --use_manifold / --manifold_dim / --vk_path from argv
    so Hydra can parse the remainder. Call before hydra.main runs.
    """
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--use_manifold", action="store_true")
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
    If --use_manifold, set use_manifold_retraction=True, load V_k from --vk_path
    or from steer_kwargs['vk_path'], optionally truncate to --manifold_dim.
    """
    out = dict(steer_kwargs)
    if not getattr(cli_args, "use_manifold", False):
        return out
    out["use_manifold"] = True
    out["use_manifold_retraction"] = True
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


def steer_name_with_manifold_suffix(base_steer_name: str, steer_model_kwargs: dict[str, Any]) -> str:
    """
    When ODE-Manifold is enabled, append ``-Manifold-k{manifold_dim}`` to the Hydra-resolved
    steer name so ``raw_outputs`` filenames and eval ``Steering Method`` differ from vanilla
    ODESteer / StepODESteer on disk (no skip / no CSV row collision).

    Manifold is considered active if ``use_manifold`` is true or a ``V_k`` tensor was injected
    (e.g. via CLI merge).
    """
    use_m = bool(steer_model_kwargs.get("use_manifold"))
    has_vk = steer_model_kwargs.get("V_k") is not None
    if not use_m and not has_vk:
        return base_steer_name
    k = int(steer_model_kwargs.get("manifold_dim", 128))
    token = f"Manifold-k{k}"
    if base_steer_name.endswith(token) or f"-{token}" in base_steer_name:
        return base_steer_name
    return f"{base_steer_name}-{token}"
