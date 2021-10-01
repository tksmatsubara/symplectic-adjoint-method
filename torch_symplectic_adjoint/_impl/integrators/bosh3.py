import torch
from .rk_common import _ButcherTableau, RKAdaptiveStepsizeODESolver


_BOGACKI_SHAMPINE_TABLEAU = _ButcherTableau(
    alpha=torch.tensor([1 / 2, 3 / 4, 1.], dtype=torch.float64),
    beta=[
        torch.tensor([1 / 2], dtype=torch.float64),
        torch.tensor([0., 3 / 4], dtype=torch.float64),
        torch.tensor([2 / 9, 1 / 3, 4 / 9], dtype=torch.float64)
    ],
    c_sol=torch.tensor([2 / 9, 1 / 3, 4 / 9, 0.], dtype=torch.float64),
    c_error=torch.tensor([2 / 9 - 7 / 24, 1 / 3 - 1 / 4, 4 / 9 - 1 / 3, -1 / 8], dtype=torch.float64),
)

_BS_C_MID = torch.tensor([0., 0.5, 0., 0.], dtype=torch.float64)

_BOGACKI_SHAMPINE_TABLEAU_NEU = _ButcherTableau(
    alpha=torch.tensor([3 / 4, 1 / 2, 0], dtype=torch.float64),
    beta=[torch.tensor([1], dtype=torch.float64),
          torch.tensor([0, 3 / 4], dtype=torch.float64)
          ],
    c_sol=torch.tensor([4 / 9, 1 / 3, 2 / 9], dtype=torch.float64),
    c_error=None
)
_BOGACKI_SHAMPINE_TABLEAU_BAR = _ButcherTableau(
    alpha=None,
    beta=[torch.tensor([0], dtype=torch.float64),
          torch.tensor([0, 0], dtype=torch.float64)
          ],
    c_sol=torch.tensor([0, 0, 0], dtype=torch.float64),
    c_error=None
)


class Bosh3Solver(RKAdaptiveStepsizeODESolver):
    order = 3
    tableau = _BOGACKI_SHAMPINE_TABLEAU
    tableau_sadjoint1_implemented = _BOGACKI_SHAMPINE_TABLEAU_NEU
    tableau_sadjoint2_implemented = _BOGACKI_SHAMPINE_TABLEAU_BAR
    mid = _BS_C_MID
