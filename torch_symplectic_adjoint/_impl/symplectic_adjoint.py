import torch
import torch.nn as nn
from .odeint import SOLVERS, odeint
from .integrators.misc import _check_inputs, _flat_to_shape, _rms_norm, _mixed_linf_rms_norm, _wrap_norm
from .integrators.rk_common import _RungeKuttaState
from .adjoint import find_parameters


class OdeintSymplecticAdjoint(torch.autograd.Function):

    @staticmethod
    def forward(ctx, shapes, func, y0, t, rtol, atol, method, options, event_fn, *adjoint_params):
        ctx.shapes = shapes
        ctx.func = func
        ctx.method = method
        ctx.rtol = rtol
        ctx.atol = atol
        ctx.options = options
        ctx.adjoint_params = adjoint_params
        solver = SOLVERS[method](func=func, y0=y0, rtol=rtol, atol=atol, checkpoint=True, **options)
        saved_solvers_mid, solver.mid = solver.mid, None
        with torch.no_grad():
            y = solver.integrate(t)
        solver.mid = saved_solvers_mid
        ctx.checkpoint = solver.checkpoint
        ctx.save_for_backward(t)
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

            checkpoint = ctx.checkpoint
            t, = ctx.saved_tensors
            assert (torch.argsort(t.cpu()) == torch.arange(len(t))).all()  # t must be ordered
            adjoint_params = tuple(adjoint_params)
            grad_y = grad_y[0]

            solver = SOLVERS[method](func=func, y0=checkpoint[0].y1, rtol=rtol, atol=atol, **options)
            solver._make_symplectic_tableau()
            saved_solvers_mid, solver.mid = solver.mid, None

            current_grad_y = grad_y[-1].clone()
            grad_params = [torch.zeros_like(p) for p in adjoint_params]
            j = len(checkpoint) - 1
            for i in range(len(t) - 1, 0, -1):
                # Run the augmented system backwards in time.
                while j > 0 and checkpoint[j - 1].t1 >= t[i - 1]:
                    y1 = checkpoint[j - 1].y1.detach()
                    t1 = checkpoint[j - 1].t1.detach()
                    dt = checkpoint[j].t1 - t1
                    current_grad_y, d_grad_params = solver._nonadaptive_step_symplectic_adjoint(y1, t1, dt, current_grad_y, adjoint_params)
                    grad_params = [
                        grad_param + d_grad_param
                        for grad_param, d_grad_param in zip(grad_params, d_grad_params)
                    ]
                    j = j - 1
                current_grad_y = current_grad_y + grad_y[i - 1]  # update any gradients wrt state at this time point
            solver.mid = saved_solvers_mid
        del checkpoint
        del ctx.checkpoint
        return (None, None, current_grad_y, None, None, None, None, None, None, *grad_params)


def odeint_symplectic_adjoint(func, y0, t, *, rtol=1e-7, atol=1e-9, method=None, options=None, event_fn=None, adjoint_params=None):

    shapes, func, y0, t, rtol, atol, method, options, event_fn, decreasing_time = _check_inputs(func, y0, t, rtol, atol, method, options, event_fn, SOLVERS)
    assert event_fn is None

    if adjoint_params is None:
        adjoint_params = tuple(find_parameters(func))
    else:
        adjoint_params = tuple(adjoint_params)  # in case adjoint_params is a generator.
    adjoint_params = tuple(p for p in adjoint_params if p.requires_grad)

    solution = OdeintSymplecticAdjoint.apply(shapes, func, y0, t, rtol, atol, method, options, event_fn, *adjoint_params)

    if shapes is not None:
        solution = _flat_to_shape(solution, (len(t),), shapes)

    return solution
