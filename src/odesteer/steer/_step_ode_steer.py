'''
Step-based CRFDSteer methods (legacy module name; legacy alias StepODESteer).
'''

from abc import abstractmethod

import torch
from torch import Tensor

from ._base_steer import Steer
from ._manifold import load_vk_from_path, norm_preserving_retraction, project_vector_field, truncate_vk
from ..utils.kernels import KernelClassifier, RFFClassifier
from ..utils.kernels import NormedPolyClassifier

_MANIFOLD_STEER_KEYS = frozenset({
    "V_k",
    "vk_path",
    "use_manifold",
    "use_manifold_retraction",
    "manifold_dim",
})


class BaseStepCRFDSteer(Steer):
    def __init__(self, **kwargs):
        super().__init__()

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

        clf_kwargs = {k: v for k, v in kwargs.items() if k not in _MANIFOLD_STEER_KEYS}
        self.clf = self._init_clf(**clf_kwargs)
        
    def fit(self, pos_X: Tensor, neg_X_or_labels: Tensor) -> 'BaseStepCRFDSteer':
        self.clf.fit(pos_X, neg_X_or_labels)
        return self
    
    @torch.no_grad()
    def steer(self, X: Tensor, T: float = 1.0) -> Tensor:
        vf = self.vector_field(X)
        X_new = X + T * vf
        if self.use_manifold_retraction:
            X_new = norm_preserving_retraction(X, X_new)
        return X_new
    
    def vector_field(self, X: Tensor) -> Tensor:
        self.clf.to(X.device)
        raw_grad = self.clf.grad(X)
        vf = raw_grad / (raw_grad.norm(dim = -1, keepdim = True) + 1e-10)
        if self.V_k is not None:
            vf = project_vector_field(vf, self.V_k)
        return vf
    
    @abstractmethod
    def _init_clf(self, **kwargs) -> KernelClassifier:
        raise NotImplementedError

        
class StepCRFDSteer(BaseStepCRFDSteer):
    '''
    One-step CRFDSteer with NormedPolyCntSketch classifier.
    '''
    def _init_clf(self, **kwargs) -> NormedPolyClassifier:
        return NormedPolyClassifier(**kwargs)
    
    
class RFFStepCRFDSteer(BaseStepCRFDSteer):
    '''
    Ablation study for one-step CRFDSteer with RFF classifier.
    '''
    def _init_clf(self, **kwargs) -> RFFClassifier:
        return RFFClassifier(**kwargs)


# Legacy names (upstream ODESteer codebase compatibility)
BaseStepODESteer = BaseStepCRFDSteer
StepODESteer = StepCRFDSteer
RFFStepODESteer = RFFStepCRFDSteer
