import torch
from .rk_common import _ButcherTableau, RKAdaptiveStepsizeODESolver


_DORMAND_PRINCE_SHAMPINE_TABLEAU = _ButcherTableau(
    alpha=torch.tensor([1 / 5, 3 / 10, 4 / 5, 8 / 9, 1., 1.], dtype=torch.float64),
    beta=[
        torch.tensor([1 / 5], dtype=torch.float64),
        torch.tensor([3 / 40, 9 / 40], dtype=torch.float64),
        torch.tensor([44 / 45, -56 / 15, 32 / 9], dtype=torch.float64),
        torch.tensor([19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729], dtype=torch.float64),
        torch.tensor([9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656], dtype=torch.float64),
        torch.tensor([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84], dtype=torch.float64),
    ],
    c_sol=torch.tensor([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0], dtype=torch.float64),
    c_error=torch.tensor([
        35 / 384 - 1951 / 21600,
        0,
        500 / 1113 - 22642 / 50085,
        125 / 192 - 451 / 720,
        -2187 / 6784 - -12231 / 42400,
        11 / 84 - 649 / 6300,
        -1. / 60.,
    ], dtype=torch.float64),
)

DPS_C_MID = torch.tensor([
    6025192743 / 30085553152 / 2, 0, 51252292925 / 65400821598 / 2, -2691868925 / 45128329728 / 2,
    187940372067 / 1594534317056 / 2, -1776094331 / 19743644256 / 2, 11237099 / 235043384 / 2
], dtype=torch.float64)

_DORMAND_PRINCE_SHAMPINE_TABLEAU_NEU = _ButcherTableau(
    alpha=torch.tensor([1, 8 / 9, 4 / 5, 3 / 10, 1 / 5, 0], dtype=torch.float64),
    beta=[
        torch.tensor([1 / 9], dtype=torch.float64),
        torch.tensor([7 / 125, 18 / 125], dtype=torch.float64),
        torch.tensor([11683 / 4500, -7049 / 1000, 371 / 72], dtype=torch.float64),
        torch.tensor([-355 / 252, 1585 / 424, -175 / 72, 75 / 742], dtype=torch.float64),
        torch.tensor([9017 / 2205, -19372 / 1855, 440 / 63, 960 / 2597, 0], dtype=torch.float64)
    ],
    c_sol=torch.tensor([11 / 84, -2187 / 6784, 125 / 192, 500 / 1113, 0, 35 / 384], dtype=torch.float64),
    c_error=None
)
_DORMAND_PRINCE_SHAMPINE_TABLEAU_BAR = _ButcherTableau(
    alpha=None,
    beta=[
        torch.tensor([0], dtype=torch.float64),
        torch.tensor([0, 0], dtype=torch.float64),
        torch.tensor([0, 0, 0], dtype=torch.float64),
        torch.tensor([0, 0, 0, 0], dtype=torch.float64),
        torch.tensor([0, 0, 0, 0, 384 / 175], dtype=torch.float64)
    ],
    c_sol=torch.tensor([0, 0, 0, 0, 1, 0], dtype=torch.float64),
    c_error=None
)


class Dopri5Solver(RKAdaptiveStepsizeODESolver):
    order = 5
    tableau = _DORMAND_PRINCE_SHAMPINE_TABLEAU
    tableau_sadjoint1_implemented = _DORMAND_PRINCE_SHAMPINE_TABLEAU_NEU
    tableau_sadjoint2_implemented = _DORMAND_PRINCE_SHAMPINE_TABLEAU_BAR
    mid = DPS_C_MID
