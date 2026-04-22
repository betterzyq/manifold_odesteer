'''
Paper: Representation Surgery: Theory and Practice of Affine Steering (ICML 2024)
Reference: 
- Paper: https://openreview.net/forum?id=GwA4go0Mw4
- Code: https://github.com/shauli-ravfogel/affine-steering
'''

import numpy as np
import torch
from torch import Tensor
from sklearn.linear_model import LogisticRegression
import ot

from ._base_steer import Steer


def _finite_np(x: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf so OT / sklearn do not propagate non-finite matrices."""
    x = np.asarray(x, dtype=np.float64)
    if np.isfinite(x).all():
        return x
    return np.nan_to_num(x, nan=0.0, posinf=1e30, neginf=-1e30)


class MiMiC(Steer):
    def __init__(
        self,
        ot_reg: float = 1e-2,
        ot_reg_schedule: tuple[float, ...] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.clf = LogisticRegression(max_iter=5000)
        self._ot_reg_default = float(ot_reg)
        # 先尝试与历史实现一致的 reg，仅在数值失败时再增大，避免无故改变 Llama/Mistral/Falcon 上已稳定的 MiMiC 行为
        self._ot_reg_schedule = ot_reg_schedule or (
            ot_reg,
            0.1,
            1.0,
            5.0,
            10.0,
            50.0,
            100.0,
        )
        self.ot_linear = ot.da.LinearTransport(reg=self._ot_reg_default)

    def fit(self, pos_X: Tensor, neg_X: Tensor):
        pos_np = _finite_np(pos_X.detach().cpu().numpy())
        neg_np = _finite_np(neg_X.detach().cpu().numpy())
        train_X = np.concatenate([pos_np, neg_np], axis=0)
        train_Y = np.concatenate(
            [np.ones(len(pos_np)), np.zeros(len(neg_np))], axis=0
        )
        self.clf.fit(train_X, train_Y)

        last_err: Exception | None = None
        for reg in self._ot_reg_schedule:
            try:
                self.ot_linear = ot.da.LinearTransport(reg=float(reg))
                self.ot_linear.fit(Xs=neg_np, Xt=pos_np)
                A = getattr(self.ot_linear, "A_", None)
                if A is not None and np.all(np.isfinite(A)):
                    B = getattr(self.ot_linear, "B_", None)
                    if B is None or np.all(np.isfinite(B)):
                        return self
                last_err = ValueError(f"non-finite OT parameters at reg={reg}")
            except (ValueError, np.linalg.LinAlgError) as e:
                last_err = e
                continue
        raise RuntimeError(
            "MiMiC: optimal-transport linear map did not yield finite parameters "
            "(try larger steer.kwargs.ot_reg / different layer, or another steer). "
            f"Last error: {last_err!r}"
        ) from last_err
    
    def steer(self, X: Tensor, T: float = 1.0):
        y_pred = self.clf.predict(X.detach().cpu().numpy())
        steered_X = X.detach().cpu().numpy().copy()
        steered_X[y_pred == 0] = self.ot_linear.transform(steered_X[y_pred == 0])
        X_new = torch.as_tensor(steered_X, device=X.device, dtype=X.dtype)
        return self._postprocess_steered(X, X_new)
    
    def vector_field(self, X: Tensor):
        X_new = self.steer(X, T=1.0)
        vf = X_new - X
        return self._postprocess_vector_field(vf)