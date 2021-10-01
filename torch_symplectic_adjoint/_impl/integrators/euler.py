import torch
from .rk_common import _ButcherTableau, RKAdaptiveStepsizeODESolver


_EULER_TABLEAU = _ButcherTableau(
    alpha=torch.tensor([0.], dtype=torch.float64),
    beta=[
    ],
    c_sol=torch.tensor([1.], dtype=torch.float64),
    c_error=None
)


class EulerSolver(RKAdaptiveStepsizeODESolver):
    order = 1
    tableau = _EULER_TABLEAU
    mid = None
