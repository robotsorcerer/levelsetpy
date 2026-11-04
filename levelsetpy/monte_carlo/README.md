# Sampling-based HJ Reachability

## Contents

- [HJ-Gauss overview](#hj-gauss-overview)
- [Animations](#animations)
- [Layout](#layout)
- [Installation and quick checks](#installation-and-quick-checks)
- [Generated results](#generated-results)
- [Numerical scope](#numerical-scope)
- [MAPF execution comparisons](../mapf/README.md)

## HJ-Gauss overview

This package implements the variable-coefficient exponential transformation used in equation B.10 of the HJ-Gauss manuscript. With elapsed time measured from the initial condition,

```text
v_t + H(t,x,Dv) = (delta/2) Delta v,
omega = exp(-c v),
omega_t = (delta/2) Delta omega + omega (R_alg + R_der).
```

The coefficient `c` is prescribed, positive, and smooth. The implementation retains the complete algebraic and derivative residual, including the spatial derivative of `1/c` in the inverse-gradient formula. The transformation is exact. The nonlinear solver may be described as a quasilinearized fixed-point iteration because each update freezes the transformed source and solves the resulting linear parabolic problem.

HJ-Gauss combines that exact transformation with Gaussian evaluation of the initial-condition and nonlinear source integrals. Its evidence workflow compares the resulting fields with independent same-viscosity solvers, checks analytic solutions, refinement, coefficient and boundary sensitivity, and binds every accepted archive to the numerical sources that produced it.

## Animations

Each animation reads left to right as **ground truth → HJ-Gauss → absolute error**.

<table>
  <tr>
    <th>Rockets</th>
    <th>Dubins vehicles</th>
  </tr>
  <tr>
    <td><img src="https://scriptedonachip.com/downloads/Papers/assets/anim/roysoc_rockets.gif" width="440" alt="HJ-Gauss rockets comparison"></td>
    <td><img src="https://scriptedonachip.com/downloads/Papers/assets/anim/roysoc_dubins.gif" width="440" alt="HJ-Gauss Dubins-vehicle comparison"></td>
  </tr>
  <tr>
    <th>Double integrator</th>
    <th>Murmurations</th>
  </tr>
  <tr>
    <td><img src="https://scriptedonachip.com/downloads/Papers/assets/anim/roysoc_double_integrator.gif" width="440" alt="HJ-Gauss double-integrator comparison"></td>
    <td><img src="https://scriptedonachip.com/downloads/Papers/assets/anim/roysoc_murmuration.gif" width="440" alt="HJ-Gauss murmuration comparison"></td>
  </tr>
</table>

## Layout

| Path | Purpose |
|---|---|
| `hjgauss/core/` | Exact transformation, spectral solver, and finite-difference solver |
| `hjgauss/validation/` | Analytic checks, game runs, refinement, evidence checks, and reports |
| `hjgauss/render/` | Publication figures, GIFs, and generated LaTeX |
| `tests/` | Transformation, solver, and provenance regression tests |

The filenames are intentionally short within these functional namespaces: for example, `core/spectral.py`, `validation/report.py`, and `render/figures.py`.

## Installation and quick checks

Run commands from this directory:

```sh
python -m pip install -e '.[dev]'
python -m pytest -q
python -m hjgauss.validation.benchmark
python -m hjgauss.render.plot --help
```

The benchmark is a deterministic one-dimensional analytic check. The unit tests and imports do not require saved multidimensional results.

## Generated results

`results/` is generated output and is excluded from version control. It need not exist before a run: every producer creates its required directories. The standard output location is `results/multidimensional/`.

The following sequence regenerates the multidimensional evidence. These are paper-grade PDE computations and can be expensive:

```sh
python -m hjgauss.validation.cases --suite analytic
python -m hjgauss.validation.fd
python -m hjgauss.validation.refine --stage refinement --workers 1
python -m hjgauss.validation.refine --stage sensitivity --workers 1
python -m hjgauss.validation.boundary \
  --cases rockets dubins_relative dubins_single double_integrator murmuration \
  --workers 1
python -m hjgauss.validation.population --n 192 --steps 800 --method fd
python -m hjgauss.validation.report
python -m hjgauss.validation.seal
```

`validation/refine.py` writes the selected resolutions to `results/multidimensional/selection.json`. The report refuses stale archives when their stored source digests no longer match the numerical producers.
It also copies the versioned, pre-run gates from `validation/criteria.json`; the report rejects altered criteria. A single worker is the conservative default because the largest three-dimensional fields are memory intensive.

Generate manuscript assets only after the native validation records pass their declared checks:

```sh
python -m hjgauss.render.figures \
  --figure-dir /path/to/HJ_Gauss/arxiv/figures
python -m hjgauss.render.population \
  --figure-dir /path/to/HJ_Gauss/arxiv/figures \
  --n 192 --steps 800 --method fd
python -m hjgauss.render.latex --paper-root /path/to/HJ_Gauss
python -m hjgauss.render.animations \
  --output /path/to/blog/downloads/Papers/assets/anim
```

The animation renderer uses matched saved times and settings; it does not rerun the PDEs.
Static error colorbars and animation error scales use compact scientific notation (for example, `6e-3`).

## Numerical scope

The multidimensional HJ-Gauss solvers retain both transformed residual components. Their deterministic grids still grow rapidly with dimension. Gaussian Duhamel sampling evaluates one frozen-source update, not the complete nonlinear iteration. The 100,000-state murmuration experiment evaluates one altitude-invariant relative-state field at a seeded cohort; it is neither a coupled 100,000-agent trajectory simulation nor a population-safety certificate. Sampled equation residuals and finite-query discrepancies support the reported numerical comparisons but do not alone provide a continuum error or safety guarantee.
