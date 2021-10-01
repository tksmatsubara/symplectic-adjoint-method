import torch
from .rk_common import _ButcherTableau, RKAdaptiveStepsizeODESolver


_MIDPOINT_TABLEAU = _ButcherTableau(
    alpha=torch.tensor([0.5], dtype=torch.float64),
    beta=[
        torch.tensor([0.5], dtype=torch.float64),
    ],
    c_sol=torch.tensor([0., 1.], dtype=torch.float64),

    c_error=None
)

class MidPointSolver(RKAdaptiveStepsizeODESolver):
    order = 2
    tableau = _MIDPOINT_TABLEAU
    mid = None
