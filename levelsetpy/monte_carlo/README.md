# Full-residual Hamilton–Jacobi computations

This package implements the variable-coefficient exponential transformation used in equation B.10 of the HJ-Gauss manuscript. With elapsed time measured from the initial condition,

```text
v_t + H(t,x,Dv) = (delta/2) Delta v,
omega = exp(-c v),
omega_t = (delta/2) Delta omega + omega (R_alg + R_der).
```

The coefficient `c` is prescribed, positive, and smooth. The implementation retains the complete algebraic and derivative residual, including the spatial derivative of `1/c` in the inverse-gradient formula. The transformation is exact. The nonlinear solver may be described as a quasilinearized fixed-point iteration because each update freezes the transformed source and solves the resulting linear parabolic problem.

## Layout

| Path | Purpose |
|---|---|
| `hjgauss/core/` | Exact transformation, spectral solver, and finite-difference solver |
| `hjgauss/validation/` | Manufactured cases, game runs, refinement, evidence checks, and reports |
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

The benchmark is a deterministic one-dimensional manufactured check. The unit tests and imports do not require saved multidimensional results.

## Generated results

`results/` is generated output and is excluded from version control. It need not exist before a run: every producer creates its required directories. The standard output location is `results/multidimensional/`.

The following sequence regenerates the multidimensional evidence. These are paper-grade PDE computations and can be expensive:

```sh
python -m hjgauss.validation.cases --suite manufactured
python -m hjgauss.validation.refine --stage refinement --workers 3
python -m hjgauss.validation.refine --stage sensitivity --workers 3
python -m hjgauss.validation.boundary \
  --cases rockets dubins_relative dubins_single double_integrator murmuration \
  --workers 3
python -m hjgauss.validation.population --n 192 --steps 800 --method fd
python -m hjgauss.validation.report
python -m hjgauss.validation.seal
```

`validation/refine.py` writes the selected resolutions to `results/multidimensional/selection.json`. The report refuses stale archives when their stored source digests no longer match the numerical producers.

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

## Numerical scope

The multidimensional solvers retain the full transformed residual. Their deterministic grids still grow rapidly with dimension. Gaussian Duhamel sampling evaluates one frozen-source update, not the complete nonlinear iteration. The 100,000-state murmuration experiment evaluates one altitude-invariant relative-state field at a seeded cohort; it is neither a coupled 100,000-agent trajectory simulation nor a population-safety certificate. Sampled equation residuals and finite-query discrepancies support the reported numerical comparisons but do not alone provide a continuum error or safety guarantee.
