"""ODE-Manifold: tangent-space projection and norm-preserving retraction."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor


def truncate_vk(Vk: Tensor, manifold_dim: int) -> Tensor:
    Vk = Vk.float().contiguous()
    if Vk.dim() != 2:
        raise ValueError(f"V_k must be 2D [d, k], got shape {tuple(Vk.shape)}")
    if Vk.shape[1] > manifold_dim:
        Vk = Vk[:, :manifold_dim].contiguous()
    return Vk


def load_vk_from_path(vk_path: str | Path, manifold_dim: int) -> Tensor:
    path = Path(vk_path)
    if not path.is_file():
        raise FileNotFoundError(f"vk_path not found: {path}")
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(loaded, dict) and "V_k" in loaded:
        loaded = loaded["V_k"]
    return truncate_vk(loaded, manifold_dim)


def project_vector_field(vf: Tensor, Vk: Tensor) -> Tensor:
    """vf: [..., d], Vk: [d, k] -> project vf onto column-space of Vk."""
    Vk = Vk.to(device=vf.device, dtype=vf.dtype)
    inner = vf @ Vk
    return inner @ Vk.mT


def norm_preserving_retraction(X: Tensor, X_new: Tensor, eps: float = 1e-8) -> Tensor:
    n0 = torch.norm(X, p=2, dim=-1, keepdim=True)
    n1 = torch.norm(X_new, p=2, dim=-1, keepdim=True)
    return X_new * (n0 / (n1 + eps))
