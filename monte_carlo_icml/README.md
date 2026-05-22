# hj-reach-sample

Sampling-based Hamilton-Jacobi reachability analysis via the quasi-linearized Cole-Hopf transformation.

This solver computes backward reachable sets (BRS) and tubes (BRT) for nonlinear dynamical systems **without a spatial grid**, using Monte Carlo Gaussian expectations. It works by transforming the viscous Hamilton-Jacobi PDE into a sequence of heat equations via a Cole-Hopf change of variables, then estimating each heat-equation solution with importance-weighted MC samples.

## Installation

```bash
pip install -e .
```

Requires Python >= 3.11, JAX, and matplotlib.

## Quick start

```python
import jax.numpy as jnp
from functools import partial
from src.config import SolverConfig
from src.hamiltonians import DoubleIntegratorHamiltonian
from src.initial_conditions import sphere_cost
from src.hj_sampler import HJReachabilitySampler

# Configure the solver
cfg = SolverConfig(delta=0.08, num_samples=8_000, t_end=1.0)

# Define the system and target set
H = DoubleIntegratorHamiltonian(u_bound=1.0)
g = partial(sphere_cost, radius=0.5)

# Evaluate the value function at query points
eval_points = jnp.zeros((100, 2))  # (M, state_dim)
sampler = HJReachabilitySampler(H, g, cfg)
values = sampler.solve(eval_points, t=0.0)
```

## API overview

### `SolverConfig`

Immutable configuration (NamedTuple):

| Field | Default | Description |
|-------|---------|-------------|
| `delta` | 0.1 | Viscosity parameter |
| `num_samples` | 10,000 | MC samples per evaluation point |
| `max_quasi_iters` | 20 | Max quasi-linearization iterations |
| `quasi_tol` | 1e-6 | Convergence tolerance |
| `time_steps` | 50 | Steps for `solve_backward` |
| `t_start` / `t_end` | 0.0 / 1.0 | Time horizon |
| `seed` | 42 | PRNG seed |

### `HJReachabilitySampler`

Main solver class.

- `solve(eval_points, t)` -- compute v(t, x) at query points (dispatches automatically)
- `solve_exact(eval_points, t)` -- exact Cole-Hopf (quadratic H only)
- `solve_quasi_linear(eval_points, t)` -- iterative quasi-linearization (general H)
- `solve_backward(eval_points)` -- full backward-time sweep, returns dict with `t`, `v`, `eval_points`

### Hamiltonians

| Class | System | State dim | `is_quadratic` |
|-------|--------|-----------|----------------|
| `QuadraticHamiltonian(dim)` | H = (1/2)\|p\|^2 | configurable | True |
| `DoubleIntegratorHamiltonian(u_bound)` | position-velocity | 2 | False |
| `DubinsHamiltonian(speed, omega_max)` | planar vehicle with heading | 3 | False |
| `RocketsRelativeHamiltonian(a, g, u_bound)` | pursuit-evasion rockets | 3 | False |

All Hamiltonians implement `__call__(t, x, p)` and are compatible with `jax.vmap` and `jax.grad`.

### Terminal costs

- `sphere_cost(x, center=None, radius=1.0)` -- signed distance to sphere
- `cylinder_cost(x, axis_align=2, center=None, radius=1.5)` -- signed distance to axis-aligned cylinder
- `quadratic_cost(x)` -- g(x) = |x|^2 (for verification)

## Examples

See [`examples/`](examples/) for runnable scripts:

- **Double integrator** -- 2D backward reachable set at multiple time snapshots
- **Dubins vehicle** -- 3D BRS with 2D heading-angle slices
- **Two-rockets pursuit-evasion** -- 3D backward reachable tube cross-sections

```bash
JAX_PLATFORMS=cpu python examples/ex_double_integrator.py
```

## Algorithm

The solver implements the approach from the ICML 2026 paper:

1. Start from the viscous HJ PDE: `v_t + H(x, Dv) = (delta/2) * laplacian(v)`
2. **Quadratic H**: exact Cole-Hopf transformation `omega = exp(-v/delta)` reduces to the heat equation, solved by MC Gaussian expectation
3. **General H**: quasi-linearize by freezing `c = 2H / (delta |Dv|^2)`, solve the resulting heat equation, update, and repeat
4. The MC estimator uses logsumexp for numerical stability and importance-weighted gradients for the co-state recovery

## Environment notes

If CUDA/cuDNN versions are mismatched, force CPU execution:

```bash
export JAX_PLATFORMS=cpu
```

## Tests

```bash
JAX_PLATFORMS=cpu python -m pytest tests/ -v
```
