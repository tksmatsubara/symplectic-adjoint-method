import torch
from .rk_common import _ButcherTableau, RKAdaptiveStepsizeODESolver


_RK38_TABLEAU = _ButcherTableau(
    alpha=torch.tensor([1 / 3, 2 / 3, 1.], dtype=torch.float64),
    beta=[
        torch.tensor([1 / 3], dtype=torch.float64),
        torch.tensor([-1 / 3, 1], dtype=torch.float64),
        torch.tensor([1, -1, 1, ], dtype=torch.float64),
    ],
    c_sol=torch.tensor([1 / 8, 3 / 8, 3 / 8, 1 / 8], dtype=torch.float64),
    c_error=None
)


class RK38Solver(RKAdaptiveStepsizeODESolver):
    order = 3
    tableau = _RK38_TABLEAU
    mid = None
