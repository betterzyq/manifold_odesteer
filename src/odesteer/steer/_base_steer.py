from abc import ABC, abstractmethod

import torch
from torch import Tensor

from ._manifold import load_vk_from_path, norm_preserving_retraction, project_vector_field, truncate_vk


class Steer(ABC):
    def __init__(self, **kwargs):
        V_k = kwargs.pop("V_k", None)
        vk_path = kwargs.pop("vk_path", None)
        use_manifold = bool(kwargs.pop("use_manifold", False))
        self.use_manifold_retraction = bool(kwargs.pop("use_manifold_retraction", False))
        manifold_dim = int(kwargs.pop("manifold_dim", 128))

        self.V_k: Tensor | None = None
        if V_k is not None:
            self.V_k = truncate_vk(V_k, manifold_dim)
        elif vk_path is not None and use_manifold:
            self.V_k = load_vk_from_path(vk_path, manifold_dim)
        if use_manifold and self.V_k is None:
            raise ValueError("use_manifold=True requires V_k or vk_path in kwargs")

        if kwargs:
            unknown = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected steer kwargs: {unknown}")
    
    @abstractmethod
    def steer(self) -> float:
        raise NotImplementedError
    
    @abstractmethod
    def vector_field(self) -> Tensor:
        raise NotImplementedError

    def _postprocess_vector_field(self, vf: Tensor) -> Tensor:
        if self.V_k is None:
            return vf
        return project_vector_field(vf, self.V_k)

    def _postprocess_steered(self, X: Tensor, X_new: Tensor) -> Tensor:
        if self.use_manifold_retraction:
            return norm_preserving_retraction(X, X_new)
        return X_new
    
    
class VecSteer(Steer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.steer_vec = None
    
    @abstractmethod
    def fit(self) -> None:
        raise NotImplementedError
    
    @torch.no_grad()
    def steer(self, X: Tensor, T: float = 1.0) -> Tensor:
        vf = self.vector_field(X)
        X_new = X + T * vf
        return self._postprocess_steered(X, X_new)
    
    def vector_field(
        self,
        X: Tensor,
    ) -> Tensor:
        vf = self.steer_vec.to(device=X.device, dtype=X.dtype).broadcast_to(X.shape)
        return self._postprocess_vector_field(vf)