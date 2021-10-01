import bisect
import collections
import torch
from .event_handling import find_event
from .interp import _interp_evaluate, _interp_fit
from .misc import (_compute_error_ratio,
                   _select_initial_step,
                   _optimal_step_size)
from .misc import Perturb
from .solvers import AdaptiveStepsizeEventODESolver


_ButcherTableau = collections.namedtuple('_ButcherTableau', 'alpha, beta, c_sol, c_error')


_RungeKuttaState = collections.namedtuple('_RungeKuttaState', 'y1, f1, t0, t1, dt, interp_coeff')
# Saved state of the Runge Kutta solver.
#
# Attributes:
#     y1: Tensor giving the function value at the end of the last time step.
#     f1: Tensor giving derivative at the end of the last time step.
#     t0: scalar float64 Tensor giving start of the last time step.
#     t1: scalar float64 Tensor giving end of the last time step.
#     dt: scalar float64 Tensor giving the size for the next time step.
#     interp_coeff: list of Tensors giving coefficients for polynomial
#         interpolation between `t0` and `t1`.


class _UncheckedAssign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scratch, value, index):
        ctx.index = index
        scratch.data[index] = value  # sneak past the version checker
        return scratch

    @staticmethod
    def backward(ctx, grad_scratch):
        return grad_scratch, grad_scratch[ctx.index], None


def _runge_kutta_step(func, y0, f0, t0, dt, t1, tableau, no_f1=False):
    """Take an arbitrary Runge-Kutta step and estimate error.
    Args:
        func: Function to evaluate like `func(t, y)` to compute the time derivative of `y`.
        y0: Tensor initial value for the state.
        f0: Tensor initial value for the derivative, computed from `func(t0, y0)`.
        t0: float64 scalar Tensor giving the initial time.
        dt: float64 scalar Tensor giving the size of the desired time step.
        t1: float64 scalar Tensor giving the end time; equal to t0 + dt. This is used (rather than t0 + dt) to ensure
            floating point accuracy when needed.
        tableau: _ButcherTableau describing how to take the Runge-Kutta step.
    Returns:
        Tuple `(y1, f1, y1_error, k)` giving the estimated function value after
        the Runge-Kutta step at `t1 = t0 + dt`, the derivative of the state at `t1`,
        estimated error at `t1`, and a list of Runge-Kutta coefficients `k` used for
        calculating these terms.
    """

    t0 = t0.to(y0.dtype)
    dt = dt.to(y0.dtype)
    t1 = t1.to(y0.dtype)

    # We use an unchecked assign to put data into k without incrementing its _version counter, so that the backward
    # doesn't throw an (overzealous) error about in-place correctness. We know that it's actually correct.
    k = torch.empty(*f0.shape, len(tableau.beta) + 1, dtype=y0.dtype, device=y0.device)
    k = _UncheckedAssign.apply(k, f0, (..., 0))
    for i, (alpha_i, beta_i) in enumerate(zip(tableau.alpha, tableau.beta)):
        if alpha_i == 1.:
            # Always step to perturbing just before the end time, in case of discontinuities.
            ti = t1
            perturb = Perturb.PREV
        else:
            ti = t0 + alpha_i * dt
            perturb = Perturb.NONE
        yi = y0 + k[..., :i + 1].matmul(beta_i * dt).view_as(f0)
        f = func(ti, yi, perturb=perturb)
        k = _UncheckedAssign.apply(k, f, (..., i + 1))

    if not (tableau.c_sol[-1] == 0 and (tableau.c_sol[:-1] == tableau.beta[-1]).all()):
        # This property (true for Dormand-Prince) lets us save a few FLOPs.
        yi = y0 + k.matmul(dt * tableau.c_sol).view_as(f0)
        f1 = func(t0 + dt, yi) if not no_f1 else None
    else:
        f1 = k[..., -1]

    y1 = yi
    y1_error = None if tableau.c_error is None else k.matmul(dt * tableau.c_error)

    return y1, f1, y1_error, k


class RKAdaptiveStepsizeODESolver(AdaptiveStepsizeEventODESolver):
    order: int
    tableau: _ButcherTableau
    mid: torch.Tensor

    def __init__(self, func, y0, rtol, atol,
                 first_step=None,
                 step_t=None,
                 jump_t=None,
                 safety=0.9,
                 ifactor=10.0,
                 dfactor=0.2,
                 max_num_steps=2 ** 31 - 1,
                 dtype=torch.float64,
                 **kwargs):
        super(RKAdaptiveStepsizeODESolver, self).__init__(dtype=dtype, y0=y0, **kwargs)

        # We use mixed precision. y has its original dtype (probably float32), whilst all 'time'-like objects use
        # `dtype` (defaulting to float64).
        dtype = torch.promote_types(dtype, y0.dtype)
        device = y0.device

        self.func = func
        self.rtol = torch.as_tensor(rtol, dtype=dtype, device=device)
        self.atol = torch.as_tensor(atol, dtype=dtype, device=device)
        self.first_step = None if first_step is None else torch.as_tensor(first_step, dtype=dtype, device=device)
        self.safety = torch.as_tensor(safety, dtype=dtype, device=device)
        self.ifactor = torch.as_tensor(ifactor, dtype=dtype, device=device)
        self.dfactor = torch.as_tensor(dfactor, dtype=dtype, device=device)
        self.max_num_steps = torch.as_tensor(max_num_steps, dtype=torch.int32, device=device)
        self.dtype = dtype

        self.step_t = None if step_t is None else torch.as_tensor(step_t, dtype=dtype, device=device)
        self.jump_t = None if jump_t is None else torch.as_tensor(jump_t, dtype=dtype, device=device)

        # Copy from class to instance to set device
        self.tableau = _ButcherTableau(alpha=self.tableau.alpha.to(device=device, dtype=y0.dtype),
                                       beta=[b.to(device=device, dtype=y0.dtype) for b in self.tableau.beta],
                                       c_sol=self.tableau.c_sol.to(device=device, dtype=y0.dtype),
                                       c_error=self.tableau.c_error.to(device=device, dtype=y0.dtype) if self.tableau.c_error is not None else None)
        self.mid = self.mid.to(device=device, dtype=y0.dtype) if self.mid is not None else None

    def _before_integrate(self, t):
        t0 = t[0]
        f0 = self.func(t[0], self.y0)
        if self.tableau.c_error is None:
            first_step = 1.
        elif self.first_step is None:
            first_step = _select_initial_step(self.func, t[0], self.y0, self.order - 1, self.rtol, self.atol,
                                              self.norm, f0=f0)
        else:
            first_step = self.first_step
        self.rk_state = _RungeKuttaState(self.y0, f0, t[0], t[0], first_step, [self.y0] * 5)

        # Handle step_t and jump_t arguments.
        if self.step_t is None:
            step_t = torch.tensor([], dtype=self.dtype, device=self.y0.device)
        else:
            step_t = _sort_tvals(self.step_t, t0)
            step_t = step_t.to(self.dtype)
        if self.jump_t is None:
            jump_t = torch.tensor([], dtype=self.dtype, device=self.y0.device)
        else:
            jump_t = _sort_tvals(self.jump_t, t0)
            jump_t = jump_t.to(self.dtype)
        counts = torch.cat([step_t, jump_t]).unique(return_counts=True)[1]
        if (counts > 1).any():
            raise ValueError("`step_t` and `jump_t` must not have any repeated elements between them.")

        self.step_t = step_t
        self.jump_t = jump_t
        self.next_step_index = min(bisect.bisect(self.step_t.tolist(), t[0]), len(self.step_t) - 1)
        self.next_jump_index = min(bisect.bisect(self.jump_t.tolist(), t[0]), len(self.jump_t) - 1)

    def _advance(self, next_t):
        """No error estimation, no adaptive stepping."""
        if self.tableau.c_error is None:
            rk_state = _RungeKuttaState(self.rk_state.y1, self.rk_state.f1, self.rk_state.t0, self.rk_state.t1, next_t - self.rk_state.t1, None)
            self.rk_state = self._nonadaptive_step(rk_state)
            if self.keep_checkpoint:
                rk_state_checkpoint = _RungeKuttaState(self.rk_state.y1, None, None, self.rk_state.t1, None, None)
                self.checkpoint.append(rk_state_checkpoint)
            return self.rk_state.y1

        """Interpolate through the next time point, integrating as necessary."""
        n_steps = 0
        while next_t > self.rk_state.t1:
            assert n_steps < self.max_num_steps, 'max_num_steps exceeded ({}>={})'.format(n_steps, self.max_num_steps)
            last_rk_state = self.rk_state
            self.rk_state = self._adaptive_step(self.rk_state, next_t)
            if self.keep_checkpoint and self.rk_state.t1 != last_rk_state.t1:
                rk_state_checkpoint = _RungeKuttaState(self.rk_state.y1, None, None, self.rk_state.t1, None, None)
                self.checkpoint.append(rk_state_checkpoint)
            n_steps += 1
        return self.rk_state.y1

    def _adaptive_step(self, rk_state, next_t=None):
        """Take an adaptive Runge-Kutta step to integrate the ODE."""
        y0, f0, _, t0, dt, interp_coeff = rk_state
        t1 = t0 + dt
        # dtypes: self.y0.dtype (probably float32); self.dtype (probably float64)
        # used for state and timelike objects respectively.
        # Then:
        # y0.dtype == self.y0.dtype
        # f0.dtype == self.y0.dtype
        # t0.dtype == self.dtype
        # dt.dtype == self.dtype
        # for coeff in interp_coeff: coeff.dtype == self.y0.dtype

        ########################################################
        #                      Assertions                      #
        ########################################################
        assert t0 + dt > t0, 'underflow in dt {}'.format(dt.item())
        assert torch.isfinite(y0).all(), 'non-finite values in state `y`: {}'.format(y0)

        ########################################################
        #     Make step, respecting prescribed grid points     #
        ########################################################

        on_step_t = False
        if len(self.step_t):
            next_step_t = self.step_t[self.next_step_index]
            on_step_t = t0 < next_step_t < t0 + dt
            if on_step_t:
                t1 = next_step_t
                dt = t1 - t0

        on_jump_t = False
        if len(self.jump_t):
            next_jump_t = self.jump_t[self.next_jump_index]
            on_jump_t = t0 < next_jump_t < t0 + dt
            if on_jump_t:
                on_step_t = False
                t1 = next_jump_t
                dt = t1 - t0

        # time step size is adapted not to exceed the terminal
        if next_t is not None:
            if next_t < t1:
                t1 = next_t
                dt = t1 - t0

        # Must be arranged as doing all the step_t handling, then all the jump_t handling, in case we
        # trigger both. (i.e. interleaving them would be wrong.)

        y1, f1, y1_error, k = _runge_kutta_step(self.func, y0, f0, t0, dt, t1, tableau=self.tableau)
        # dtypes:
        # y1.dtype == self.y0.dtype
        # f1.dtype == self.y0.dtype
        # y1_error.dtype == self.dtype
        # k.dtype == self.y0.dtype

        ########################################################
        #                     Error Ratio                      #
        ########################################################
        error_ratio = _compute_error_ratio(y1_error, self.rtol, self.atol, y0, y1, self.norm)
        accept_step = error_ratio <= 1
        # dtypes:
        # error_ratio.dtype == self.dtype

        ########################################################
        #                   Update RK State                    #
        ########################################################
        if accept_step:
            t_next = t1
            y_next = y1
            interp_coeff = self._interp_fit(y0, y_next, k, dt)
            if on_step_t:
                if self.next_step_index != len(self.step_t) - 1:
                    self.next_step_index += 1
            if on_jump_t:
                if self.next_jump_index != len(self.jump_t) - 1:
                    self.next_jump_index += 1
                # We've just passed a discontinuity in f; we should update f to match the side of the discontinuity
                # we're now on.
                f1 = self.func(t_next, y_next, perturb=Perturb.NEXT)
            f_next = f1
        else:
            t_next = t0
            y_next = y0
            f_next = f0
        if y1_error is not None:
            dt_next = _optimal_step_size(dt, error_ratio, self.safety, self.ifactor, self.dfactor, self.order)
        else:
            dt_next = dt
        rk_state = _RungeKuttaState(y_next, f_next, t0, t_next, dt_next, interp_coeff)
        return rk_state

    def _nonadaptive_step(self, rk_state, no_f1=False):
        """Take a non-adaptive Runge-Kutta step to integrate the ODE."""
        y0, f0, _, t0, dt, _ = rk_state
        t1 = t0 + dt
        y1, f1, _, _ = _runge_kutta_step(self.func, y0, f0, t0, dt, t1, tableau=self.tableau, no_f1=no_f1)
        rk_state = _RungeKuttaState(y1, f1, t0, t1, dt, None)
        return rk_state

    def _nonadaptive_step_symplectic_adjoint(self, y0, t0, dt, current_grad_y, adjoint_params, is64=False):
        """Take a non-adaptive Runge-Kutta step to integrate the ODE with all stages."""
        # =================
        # GET INTERMEDIATE STEPS
        # =================

        func = self.func
        t0 = t0.to(y0.dtype)
        dt = dt.to(y0.dtype)
        yis = [y0]

        if self.has_redundant_step:
            alpha = self.tableau.alpha[:-1]
            beta = self.tableau.beta[:-1]
        else:
            alpha = self.tableau.alpha
            beta = self.tableau.beta

        # We use an unchecked assign to put data into k without incrementing its _version counter, so that the backward
        # doesn't throw an (overzealous) error about in-place correctness. We know that it's actually correct.
        f0 = func(t0, y0)
        k = torch.empty(*f0.shape, len(beta) + 1, dtype=y0.dtype, device=y0.device)
        k = _UncheckedAssign.apply(k, f0, (..., 0))
        for i, (alpha_i, beta_i) in enumerate(zip(alpha, beta)):
            if alpha_i == 1.:
                # Always step to perturbing just before the end time, in case of discontinuities.
                ti = t0 + dt
                perturb = Perturb.PREV
            else:
                ti = t0 + alpha_i * dt
                perturb = Perturb.NONE
            yi = y0 + k[..., :i + 1].matmul(beta_i * dt).view_as(y0)
            yis.append(yi)
            if i < len(beta) - 1:
                f = func(ti, yi, perturb=perturb)
                k = _UncheckedAssign.apply(k, f, (..., i + 1))
        del k, alpha, beta, yi

        # =================
        # RUN SYMPLECTIC ADJOINT EQUATION
        # =================
        if is64:
            grad_params = [torch.zeros_like(p).type(torch.float64) for p in adjoint_params]
        else:
            grad_params = [torch.zeros_like(p) for p in adjoint_params]

        lam1 = current_grad_y.clone()
        yis = yis[::-1]

        alpha = self.tableau_sadjoint1.alpha
        beta = [b_neu + dt * b_bar for b_neu, b_bar in zip(self.tableau_sadjoint1.beta, self.tableau_sadjoint2.beta)]
        c_sol = self.tableau_sadjoint1.c_sol + dt * self.tableau_sadjoint2.c_sol
        c_sol_neu = self.tableau_sadjoint1.c_sol

        def adjoint_eq(func, y, t, l, params):
            y = y.detach().requires_grad_(True)
            with torch.enable_grad():
                f = func(t, y)
            grad_y_i, *grad_params_i = torch.autograd.grad(f, (y,) + params, -l)
            return grad_y_i, grad_params_i

        def update_adjoint_param(grad_params, d_grad_params, b):
            if is64:
                return [p + b.type(torch.float64) * g.type(torch.float64) for p, g in zip(grad_params, d_grad_params)]
            else:
                return [p + b * g for p, g in zip(grad_params, d_grad_params)]

        l = torch.empty(*lam1.shape, len(beta) + 1, dtype=lam1.dtype, device=lam1.device)
        d_lam, d_param = adjoint_eq(func, yis[0], t0 + alpha[0] * dt, lam1, adjoint_params)
        grad_params = update_adjoint_param(grad_params, d_param, (-dt) * c_sol[0])
        l = _UncheckedAssign.apply(l, d_lam, (..., 0))

        for i, (alpha_i, beta_i) in enumerate(zip(alpha[1:], beta)):
            if c_sol_neu[i + 1] != 0.:
                lam_i = lam1 + l[..., :i + 1].matmul((-dt) * beta_i).view_as(d_lam)
            else:
                lam_i = l[..., :i + 1].matmul(-beta_i).view_as(d_lam)

            ti = t0 if alpha_i == 0. else t0 + alpha_i * dt
            d_lam, d_param = adjoint_eq(func, yis[i + 1], ti, lam_i, adjoint_params)
            grad_params = update_adjoint_param(grad_params, d_param, (-dt) * c_sol[i + 1])
            l = _UncheckedAssign.apply(l, d_lam, (..., i + 1))
        lam0 = lam1 + l.matmul((-dt) * c_sol).view_as(d_lam)

        return lam0, grad_params

    def _interp_fit(self, y0, y1, k, dt):
        """Fit an interpolating polynomial to the results of a Runge-Kutta step."""
        if self.mid is None:
            return None
        dt = dt.type_as(y0)
        y_mid = y0 + k.matmul(dt * self.mid).view_as(y0)
        f0 = k[..., 0]
        f1 = k[..., -1]
        return _interp_fit(y0, y1, y_mid, f0, f1, dt)

    def _make_symplectic_tableau(self):
        if hasattr(self, 'tableau_sadjoint1'):
            return
        self.has_redundant_step = self.tableau.c_sol[-1] == 0

        if hasattr(self, 'tableau_sadjoint1_implemented'):
            device = self.tableau.alpha.device
            dtype = self.tableau.alpha.dtype
            self.tableau_sadjoint1 = _ButcherTableau(
                alpha=self.tableau_sadjoint1_implemented.alpha.to(device=device, dtype=dtype),
                beta=[b.to(device=device, dtype=dtype) for b in self.tableau_sadjoint1_implemented.beta],
                c_sol=self.tableau_sadjoint1_implemented.c_sol.to(device=device, dtype=dtype),
                c_error=None)
            self.tableau_sadjoint2 = _ButcherTableau(
                alpha=None,
                beta=[b.to(device=device, dtype=dtype) for b in self.tableau_sadjoint2_implemented.beta],
                c_sol=self.tableau_sadjoint2_implemented.c_sol.to(device=device, dtype=dtype),
                c_error=None)
            return

        if self.has_redundant_step:
            alpha_neu = self.tableau.alpha[:-1].clone()
            beta_neu = [torch.zeros_like(b) for b in self.tableau.beta[:-1]]
            c_sol_neu = self.tableau.c_sol[:-1].clone()
        else:
            alpha_neu = self.tableau.alpha.clone()
            beta_neu = [torch.zeros_like(b) for b in self.tableau.beta]
            c_sol_neu = self.tableau.c_sol.clone()

        alpha_neu = torch.cat([torch.zeros_like(alpha_neu[:1]), alpha_neu], dim=0).flip(0)

        c_sol_bar = torch.zeros_like(c_sol_neu)
        c_sol_bar[c_sol_neu == 0] = 1.

        s = len(beta_neu)
        beta_bar = [torch.zeros_like(b) for b in beta_neu]
        for j in range(s):
            for i in range(0, j + 1):
                if c_sol_neu[i] != 0.:
                    if c_sol_neu[j + 1] != 0.:
                        beta_neu[s - i - 1][s - j - 1] = self.tableau.c_sol[j + 1] * self.tableau.beta[j][i] / self.tableau.c_sol[i]
                    else:
                        beta_bar[s - i - 1][s - j - 1] = self.tableau.beta[j][i] / self.tableau.c_sol[i]
                else:
                    if self.tableau.c_sol[j + 1] != 0.:
                        beta_neu[s - i - 1][s - j - 1] = self.tableau.c_sol[j + 1] * self.tableau.beta[j][i]
                    else:
                        beta_bar[s - i - 1][s - j - 1] = self.tableau.beta[j][i]

        c_sol_neu = c_sol_neu.flip(0)
        c_sol_bar = c_sol_bar.flip(0)
        self.tableau_sadjoint1 = _ButcherTableau(
            alpha=alpha_neu,
            beta=beta_neu,
            c_sol=c_sol_neu,
            c_error=None)
        self.tableau_sadjoint2 = _ButcherTableau(
            alpha=alpha_neu,
            beta=beta_bar,
            c_sol=c_sol_bar,
            c_error=None)


def _sort_tvals(tvals, t0):
    # TODO: add warning if tvals come before t0?
    tvals = tvals[tvals >= t0]
    return torch.sort(tvals).values
