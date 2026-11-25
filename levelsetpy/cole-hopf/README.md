# Cole--Hopf Transformed Solvers for Viscous HJ Reachability

Paper: [Cole--Hopf Transformed Solvers for Viscous Hamilton--Jacobi Reachability](https://scriptedonachip.com/downloads/Papers/colehopf.pdf)

## Objective

This project implements the deterministic numerical work behind the Cole--Hopf paper. For

```text
v_t + H(t, x, Dv) = (delta/2) Delta v,
omega = exp(-c v),
```

with a prescribed positive, smooth coefficient `c(t, x)`, the code evolves

```text
omega_t = (delta/2) Delta omega + omega (R_alg + R_der).
```

Both residual terms are retained. The inverse-gradient formula includes the spatial derivative of `1/c`; dropping it or `R_der` changes the equation. The objective is to verify this exact transformation numerically, compare recovered fields with independent discretizations of the same viscous equation, and bind every accepted result to the source that produced it.

The nested Monte Carlo Picard evaluator from the companion sampling paper is intentionally separate under [`../hjgauss/`](../hjgauss/).

## Layout

| Path | Purpose |
|---|---|
| `colehopf/core/transform.py` | One-dimensional transform identities, inverse map, deterministic periodic solve, and one frozen-source Gaussian Duhamel diagnostic |
| `colehopf/core/spectral.py` | Multidimensional Fourier/exponential transformed solver and independent finite-difference reference |
| `colehopf/core/finite_diff.py` | Fourth-order transformed solver and sixth-order original-equation reference |
| `colehopf/validation/` | Analytic cases, game runs, refinement, sensitivity, evidence checks, reports, and archive sealing |
| `colehopf/render/` | Static figures, animations, composite renderers, and generated LaTeX |
| `tests/` | Chain-rule, refinement, solver, and provenance regression tests |

## Reported results

The paper's selected multidimensional runs compare the transformed solver with an independent solver at the same viscosity. The reported maximum discrepancies are:

| Case | Maximum absolute discrepancy |
|---|---:|
| Double integrator | `5.6024e-4` |
| Single Dubins vehicle | `1.7754e-3` |
| Relative Dubins game | `2.4724e-3` |
| Rockets game | `3.2232e-3` |
| Starling-inspired geometry | `2.9126e-3` |

The validation workflow also checks analytic solutions, temporal and spatial refinement, coefficient sensitivity, boundary sensitivity, sampled equation residuals, positivity of the transformed field, fixed acceptance criteria, and source/archive hashes.

These comparisons are evidence for the implemented finite-grid solvers. The independent reference solves the same viscous PDE; it is not the unknown continuum solution, and the sampled residual checks are not global error bounds.

## Install and test

Run from this directory:

```sh
python -m pip install -e '.[dev]'
python -m pytest -q
python -m colehopf.validation.benchmark
```

Install the optional rendering dependencies when rebuilding animations or MuJoCo composites:

```sh
python -m pip install -e '.[dev,render]'
```

The benchmark is a deterministic one-dimensional analytic diagnostic. Tests do not require saved multidimensional archives.

## Reproduce the multidimensional evidence

The paper-grade computations can be expensive. Generated output is written below the ignored `results/` directory.

```sh
python -m colehopf.validation.cases --suite analytic
python -m colehopf.validation.fd
python -m colehopf.validation.refine --stage refinement --workers 1
python -m colehopf.validation.refine --stage sensitivity --workers 1
python -m colehopf.validation.boundary \
  --cases rockets dubins_relative dubins_single double_integrator murmuration \
  --workers 1
python -m colehopf.validation.population --n 192 --steps 800 --method fd
python -m colehopf.validation.report
python -m colehopf.validation.seal
```

`validation/refine.py` records the selected resolutions in `results/multidimensional/selection.json`. The report rejects changed acceptance criteria, stale source digests, and invalid or tampered archives. A single worker is the conservative default for the largest three-dimensional fields.

## Publication assets

After the native validation records pass:

```sh
python -m colehopf.render.figures --figure-dir /path/to/HJ_Gauss/arxiv/figures
python -m colehopf.render.population \
  --figure-dir /path/to/HJ_Gauss/arxiv/figures \
  --n 192 --steps 800 --method fd
python -m colehopf.render.latex --paper-root /path/to/HJ_Gauss
python -m colehopf.render.animations \
  --output /path/to/blog/downloads/Papers/assets/anim
```

The renderers consume saved native results; they do not silently rerun the PDEs. See [`colehopf/render/README.md`](colehopf/render/README.md) for the matched vehicle-composite workflow.

## Scope

The deterministic tensor grids still grow rapidly with state dimension. The seeded 100,000-state starling query evaluates one altitude-invariant relative-state field at a cohort of query points; it is not a coupled 100,000-agent trajectory simulation or a population-wide safety certificate.
