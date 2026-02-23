# HJ Reachability Sampling Solver — Handoff Report

## What This Is

A pure JAX solver for viscous Hamilton-Jacobi PDEs via Cole-Hopf transformation + Monte Carlo Gaussian expectations. No grids, no neural nets. Validated against levelsetpy's grid-based Lax-Friedrichs solver.

## Mathematical Foundation

**Problem**: Solve `v_t + H(t,x,Dv) = (delta/2) * Delta v` (viscous HJ PDE).

**Key insight**: The Cole-Hopf transform `omega = exp(-c*v)` converts this to a heat equation **only when c is constant** (i.e., `H = (1/2)|p|^2`). For general H, the transform produces a residual `R` involving derivatives of `c = 2H/(delta|Dv|^2)`.

**Approach**: Quasi-linearization — freeze c at the current iterate, solve the resulting heat equation via MC, update, repeat.

**Core formulas**:
- Value: `v(t,x) = -(1/c) * log E[exp(-c * g(x + sigma*z))]`, `z ~ N(0,I)`, `sigma = sqrt(delta*(T-t))`
- Gradient: `Dv = E[Dg(y)*exp(-c*g(y))] / E[exp(-c*g(y))]`

## Proof Verification (ICML26 Paper)

5 issues found in the paper's proofs:
1. **Prop 3.1 typo**: RHS should be 0, not H
2. **Non-constant coefficient** (HIGH): `c = 2H/(delta|Dv|^2)` depends on solution — not exact linearization, only quasi-linearization
3. **Prefactor drop**: `|Dg|^2/H` in Eqs (11)→(12) valid only at t=0
4. **Sign error**: Possible issue in (B.11a)
5. **Non-self-contained**: Lemmas 3.3-3.4 depend on unknown higher derivatives

Full corrected derivation in `tex/re_derivation.tex`.

## Code Architecture

```
src/
├── config.py            # SolverConfig NamedTuple (JAX PyTree-compatible)
├── transforms.py        # cole_hopf_forward/inverse, compute_frozen_coefficient
├── heat_solver.py       # MC kernels: mc_value_at_point, mc_gradient_at_point + batch wrappers
├── hj_sampler.py        # HJReachabilitySampler: solve(), solve_exact(), solve_quasi_linear(), solve_backward()
├── diagnostics.py       # compute_error_metrics(), convergence_rate()
├── initial_conditions.py # sphere_cost, cylinder_cost, quadratic_cost
└── hamiltonians/
    ├── base.py           # Abstract Hamiltonian ABC
    ├── quadratic.py      # H = (1/2)|p|^2 (exact Cole-Hopf)
    ├── double_integrator.py  # H = -p1*x2 - u*|p2| (dim=2)
    └── rockets_relative.py   # Two rockets min-max (dim=3)
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| `NamedTuple` config (not dataclass) | JAX PyTree-compatible, immutable |
| `@jax.jit` on MC kernels | Largest compilable unit for performance |
| `jax.vmap` over eval points | Batch without Python loops |
| Logsumexp stability | Mandatory — exponentials overflow otherwise |
| Smoothed `\|z\| = sqrt(z^2 + eps^2)` | JAX needs differentiable abs for `jax.grad` |
| Explicit PRNG splitting | JAX requires — never reuse keys |
| No grid storage | Value function evaluated on-demand at query points |

## Test Suite (78 tests passing)

```
tests/
├── conftest.py          # Fixtures: rng, default_config, tight_config
├── test_transforms.py   # 13 tests: roundtrip, forward, inverse, frozen coeff
├── test_heat_solver.py  # 11 tests: constant/linear/quadratic cost, gradient, batch
├── test_hamiltonians.py # 18 tests: quadratic, double integrator, rockets
├── test_hj_sampler.py   # 11 tests: exact, backward, quasi-linear (dint + rockets)
├── test_validation.py   # 10 tests: levelsetpy cross-validation, analytical, MC convergence
└── test_load.py         # 15 tests: large J, many iters, variance, edge cases, backward soak
```

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate 311 && JAX_PLATFORMS=cpu python -m pytest tests/ -v`

**GPU note**: cuDNN version mismatch (9.7.1 vs 9.8.0) — use `JAX_PLATFORMS=cpu` until fixed.

## Completion Status

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | DONE | Config, transforms, Hamiltonian ABC, initial conditions |
| 2 | DONE | Core MC heat kernel, exact Cole-Hopf solver |
| 3 | DONE | Double integrator + rockets Hamiltonians |
| 4 | DONE | Quasi-linearization iteration loop |
| 5 | DONE | Validation vs levelsetpy (10 tests: structural, quantitative, analytical, convergence) |
| 6 | DONE | Load/soak testing (15 tests: large J, many iters, variance, edge cases, backward soak) |
| 7 | DONE | TeX re-derivation document |

## Remaining Work

- **Phase 5**: Run `test_validation.py`, fix any failures. Tests compare viscous (ours, delta>0) vs inviscid (levelsetpy, delta=0) — expect O(sqrt(delta)) discrepancy per Crandall-Lions.
- **Phase 6**: `test_load.py` — J=1M no OOM, 100 quasi-linear iters no drift, deterministic seeds, CPU/GPU parity.
- **Notebooks**: `01_exact_cole_hopf.ipynb`, `02_double_integrator.ipynb`, `03_rockets.ipynb`, `04_error_analysis.ipynb`
- **Visualization module**: `src/visualization.py` (zero level set plots)

## Important Correction: Quadratic Cost Analytical Formula

For `g(x) = |x|^2` with `H = (1/2)|p|^2`, the exact Cole-Hopf MC formula gives:

```
v(t,x) = |x|^2 / (1 + 2*tau) + (n*delta/2) * log(1 + 2*tau)
```

where `tau = T - t`. This is NOT `|x|^2 + n*delta*(T-t)`. The denominator `(1+2*tau)` comes from the Gaussian moment generating function of the quadratic exponent `exp(-c*|x+sigma*z|^2)`.

## Critical Context for Continuation

- Conda env: `311` (Python 3.11, JAX installed)
- levelsetpy path: `/home/lex/Documents/ML-Control-Rob/control/levelsetpy` (must be on `sys.path`)
- levelsetpy uses **torch** + **numpy**; our solver uses **JAX** — no mixing within computation
- The `HJReachabilitySampler.solve()` auto-dispatches: quadratic H → exact, general H → quasi-linear
- `mc_value_batch` accepts scalar or per-point `c` array — quasi-linear path passes per-point frozen c
