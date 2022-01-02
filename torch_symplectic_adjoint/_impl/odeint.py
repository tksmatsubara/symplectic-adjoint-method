from .integrators.dopri8 import Dopri8Solver
from .integrators.dopri5 import Dopri5Solver
from .integrators.bosh3 import Bosh3Solver
from .integrators.fehlberg2 import Fehlberg2Solver
from .integrators.adaptive_heun import AdaptiveHeunSolver
from .integrators.misc import _check_inputs, _flat_to_shape

from .integrators.rk38 import RK38Solver
from .integrators.rk4 import RK4Solver
from .integrators.midpoint import MidPointSolver
from .integrators.euler import EulerSolver

SOLVERS = {
    'euler': EulerSolver,
    'midpoint': MidPointSolver,
    'rk38': RK38Solver,
    'rk4': RK4Solver,
    'dopri8': Dopri8Solver,
    'dopri5': Dopri5Solver,
    'bosh3': Bosh3Solver,
    'fehlberg2': Fehlberg2Solver,
    'adaptive_heun': AdaptiveHeunSolver,
}


def odeint(func, y0, t, *, rtol=1e-7, atol=1e-9, method=None, options=None, event_fn=None):
    """Integrate a system of ordinary differential equations.

    Solves the initial value problem for a non-stiff system of first order ODEs:
        ```
        dy/dt = func(t, y), y(t[0]) = y0
        ```
    where y is a Tensor or tuple of Tensors of any shape.

    Output dtypes and numerical precision are based on the dtypes of the inputs `y0`.

    Args:
        func: Function that maps a scalar Tensor `t` and a Tensor holding the state `y`
            into a Tensor of state derivatives with respect to time. Optionally, `y`
            can also be a tuple of Tensors.
        y0: N-D Tensor giving starting value of `y` at time point `t[0]`. Optionally, `y0`
            can also be a tuple of Tensors.
        t: 1-D Tensor holding a sequence of time points for which to solve for
            `y`, in either increasing or decreasing order. The first element of
            this sequence is taken to be the initial time point.
        rtol: optional float64 Tensor specifying an upper bound on relative error,
            per element of `y`.
        atol: optional float64 Tensor specifying an upper bound on absolute error,
            per element of `y`.
        method: optional string indicating the integration method to use.
        options: optional dict of configuring options for the indicated integration
            method. Can only be provided if a `method` is explicitly set.
        event_fn: Function that maps the state `y` to a Tensor. The solve terminates when
            event_fn evaluates to zero. If this is not None, all but the first elements of
            `t` are ignored.

    Returns:
        y: Tensor, where the first dimension corresponds to different
            time points. Contains the solved value of y for each desired time point in
            `t`, with the initial value `y0` being the first element along the first
            dimension.

    Raises:
        ValueError: if an invalid `method` is provided.
    """

    assert event_fn is None

    shapes, func, y0, t, rtol, atol, method, options, event_fn, t_is_reversed = _check_inputs(func, y0, t, rtol, atol, method, options, event_fn, SOLVERS)

    solver = SOLVERS[method](func=func, y0=y0, rtol=rtol, atol=atol, **options)

    solution = solver.integrate(t)

    if shapes is not None:
        solution = _flat_to_shape(solution, (len(t),), shapes)

    return solution
