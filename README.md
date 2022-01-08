# PyTorch Implementation of Symplectic Adjoint Method

A neural network model of a differential equation, namely, neural ODE, has enabled the learning of continuous-time dynamical systems and probabilistic distributions with high accuracy. The neural ODE uses the same network repeatedly during numerical integration. The memory consumption of the backpropagation algorithm is proportional to the number of uses *times* the network size. This is true even if a checkpointing scheme divides the computational graph into sub-graphs. Otherwise, the adjoint method obtains a gradient by a numerical integration backward in time. Although this method consumes memory only for a single network use, it requires high computational cost to suppress numerical errors. This study proposes the symplectic adjoint method, which is an adjoint method solved by a symplectic integrator. The symplectic adjoint method obtains the exact gradient (up to rounding error) with memory proportional to the number of uses *plus* the network size. The experimental results demonstrate that the symplectic adjoint method consumes much less memory than the naive backpropagation algorithm and checkpointing schemes, performs faster than the adjoint method, and is more robust to rounding errors.

| Methods                         | Implementation              | Gradient calculation      | Exact gradient | Memory for checkpoints | Memory for backprop. | Computational Cost |
| :------------------------------ | :-------------------------- | :------------------------ | :------------- | :--------------------- | -------------------- | ------------------ |
| Neural ODE (Chen+, NeurIPS2018) | `odeint_adjoint`            | adjoint method            | no             | O(M)                   | O(L)                 | O((N+2N')sLM)      |
| Neural ODE (Chen+, NeurIPS2018) | `odeint`                    | backpropagation           | yes            | ---                    | O(NsLM)              | O(2NsLM)           |
| baseline                        | `odeint_onecheckpoint`      | backpropagation           | yes            | O(M)                   | O(NsL)               | O(3NsLM)           |
| ACA (Zhuang+, ICML2020)         | `odeint_checkpoint`         | backpropagation           | yes            | O(MN)                  | O(sL)                | O(3NsLM)           |
| **proposed**                    | `odeint_symplectic_adjoint` | symplectic adjoint method | yes            | O(MN+s)                | O(L)                 | O(4NsLM)           |

- M: the number of stacked neural ODE components.
- L: the number of layers in a neural network.
- N, N': the number of time steps in the forward and backward integrations, respectively.
- s: the number of uses of a neural network $f$ per step.

## Installation

```bash
python setup.py install
```

## Requirements

- Python v3.7.3
- PyTorch v1.7.1

For a newer version of PyTorch, see the [beta branch](https://github.com/tksmatsubara/symplectic-adjoint-method/tree/beta).

## Usage

This library extends the implementation of neural ODE, [torchdiffeq](https://github.com/rtqichen/torchdiffeq) 0.1.1. Please refer to the repository for basic usage. Each of the following commands solves an initial value problem. The difference lies in the algorithm for obtaining the gradient.

```python
from torch_symplectic_adjoint import odeint
odeint(func, y0, t)

from torch_symplectic_adjoint import odeint_adjoint
odeint_adjoint(func, y0, t)

from torch_symplectic_adjoint import odeint_onecheckpoint
odeint_onecheckpoint(func, y0, t)

from torch_symplectic_adjoint import odeint_checkpoint
odeint_checkpoint(func, y0, t)

from torch_symplectic_adjoint import odeint_symplectic_adjoint
odeint_symplectic_adjoint(func, y0, t)
```

Options:

- `rtol` Relative tolerance.
- `atol` Absolute tolerance.
- `method` One of the solvers listed below.

Adaptive-step:

- `dopri8` Runge-Kutta 7(8) of Dormand-Prince-Shampine
- `dopri5` Runge-Kutta 4(5) of Dormand-Prince
- `bosh3` Runge-Kutta 2(3) of Bogacki-Shampine
- `fehlberg2` Runge-Kutta 1(2) of Fehlberg
- `adaptive_heun` Runge-Kutta 1(2)

Fixed-step:

- `rk38` Fourth-order Runge-Kutta with 3/8 rule.
- `rk4` The original fourth-order Runge-Kutta.
- `midpoint` Explicit midpoint method.
- `euler` Euler method.

## Reference

```bibtex
@inproceedings{Matsubara2021,
  title = {Symplectic Adjoint Method for Exact Gradient of Neural ODE with Minimal Memory},
  author = {Takashi Matsubara, Yuto Miyatake, and Takaharu Yaguchi},
  booktitle = {Advances in Neural Information Processing Systems 34 (NeurIPS2021)},
  url={https://arxiv.org/abs/2102.09750},
  year = {2021}
}
```