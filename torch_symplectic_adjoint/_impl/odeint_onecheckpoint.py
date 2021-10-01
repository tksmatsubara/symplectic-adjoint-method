import torch
import torch.nn as nn
from .odeint import SOLVERS, odeint
from .integrators.misc import _check_inputs, _flat_to_shape, _rms_norm, _mixed_linf_rms_norm, _wrap_norm
from .integrators.rk_common import _RungeKuttaState
from .adjoint import find_parameters


class OdeintOneCheckpoint(torch.autograd.Function):

    @staticmethod
    def forward(ctx, shapes, func, y0, t, rtol, atol, method, options, event_fn, *adjoint_params):
        ctx.shapes = shapes
        ctx.func = func
        ctx.method = method
        ctx.rtol = rtol
        ctx.atol = atol
        ctx.options = options
        ctx.adjoint_params = adjoint_params
        with torch.no_grad():
            solver = SOLVERS[method](func=func, y0=y0, rtol=rtol, atol=atol, **options)
            y = solver.integrate(t)
        ctx.save_for_backward(y0, t)
        return y

    @staticmethod
    def backward(ctx, *grad_y):
        with torch.no_grad():
            shapes = ctx.shapes
            func = ctx.func
            method = ctx.method
            rtol = ctx.rtol
            atol = ctx.atol
            options = ctx.options
            adjoint_params = ctx.adjoint_params
            y0, t = ctx.saved_tensors

            adjoint_params = tuple(adjoint_params)
            grad_y = grad_y[0].clone()
            y0 = y0.requires_grad_(True)

        with torch.enable_grad():
            solver = SOLVERS[method](func=func, y0=y0, rtol=rtol, atol=atol, **options)
            y = solver.integrate(t)

        with torch.no_grad():
            grad_y, *grad_adjoint_params = torch.autograd.grad(
                y, (y0, *adjoint_params), grad_y, allow_unused=True,
            )

        return (None, None, grad_y, None, None, None, None, None, None, *grad_adjoint_params)


def odeint_onecheckpoint(func, y0, t, *, rtol=1e-7, atol=1e-9, method=None, options=None, event_fn=None, adjoint_params=None):

    shapes, func, y0, t, rtol, atol, method, options, event_fn, decreasing_time = _check_inputs(func, y0, t, rtol, atol, method, options, event_fn, SOLVERS)
    assert event_fn is None

    if adjoint_params is None:
        adjoint_params = tuple(find_parameters(func))
    else:
        adjoint_params = tuple(adjoint_params)  # in case adjoint_params is a generator.
    adjoint_params = tuple(p for p in adjoint_params if p.requires_grad)

    solution = OdeintOneCheckpoint.apply(shapes, func, y0, t, rtol, atol, method, options, event_fn, *adjoint_params)

    if shapes is not None:
        solution = _flat_to_shape(solution, (len(t),), shapes)

    return solution
