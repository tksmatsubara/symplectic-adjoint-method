import torch
from .rk_common import _ButcherTableau, RKAdaptiveStepsizeODESolver


_RK4_TABLEAU = _ButcherTableau(
    alpha=torch.tensor([1 / 2, 1 / 2, 1.], dtype=torch.float64),
    beta=[
        torch.tensor([1 / 2], dtype=torch.float64),
        torch.tensor([0, 1 / 2], dtype=torch.float64),
        torch.tensor([0, 0, 1], dtype=torch.float64),
    ],
    c_sol=torch.tensor([1 / 6, 1 / 3, 1 / 3, 1 / 6], dtype=torch.float64),
    c_error=None
)


class RK4Solver(RKAdaptiveStepsizeODESolver):
    order = 3
    tableau = _RK4_TABLEAU
    mid = None
