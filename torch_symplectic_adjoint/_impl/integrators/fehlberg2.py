import torch
from .rk_common import _ButcherTableau, RKAdaptiveStepsizeODESolver


_FEHLBERG_TABLEAU = _ButcherTableau(
    alpha=torch.tensor([1 / 2, 1.], dtype=torch.float64),
    beta=[
        torch.tensor([1 / 2], dtype=torch.float64),
        torch.tensor([1 / 256, 255 / 256], dtype=torch.float64),
    ],
    c_sol=torch.tensor([1 / 512, 255 / 256, 1 / 512], dtype=torch.float64),
    c_error=torch.tensor([1 / 256, 255 / 256, 0], dtype=torch.float64),
)

_FEHLBERG_TABLEAU_NEU = _ButcherTableau(
    alpha=torch.tensor([1.0, 1 / 2, 0], dtype=torch.float64),
    beta=[torch.tensor([1 / 512], dtype=torch.float64),
          torch.tensor([1 / 256, 255 / 1], dtype=torch.float64)
          ],
    c_sol=torch.tensor([1 / 512, 255 / 256, 1 / 512], dtype=torch.float64),
    c_error=None
)
_FEHLBERG_TABLEAU_BAR = _ButcherTableau(
    alpha=None,
    beta=[torch.tensor([0], dtype=torch.float64),
          torch.tensor([0, 0], dtype=torch.float64)
          ],
    c_sol=torch.tensor([0, 0, 0], dtype=torch.float64),
    c_error=None
)


class Fehlberg2Solver(RKAdaptiveStepsizeODESolver):
    order = 2
    tableau = _FEHLBERG_TABLEAU
    tableau_sadjoint1_implemented = _FEHLBERG_TABLEAU_NEU
    tableau_sadjoint2_implemented = _FEHLBERG_TABLEAU_BAR
    mid = None
