# Full residual Hamilton–Jacobi computations

This directory implements the generalized Cole–Hopf transformation with the complete residual in equation B.10 of the HJ-Gauss manuscript. For elapsed time from the initial condition,

```text
v_t + H(t,x,Dv) = (delta/2) Delta v,
omega = exp(-c v),
omega_t = (delta/2) Delta omega + omega (R_alg + R_der).
```

The prescribed coefficient is positive and smooth. Both residual terms are retained. The inverse gradient includes the spatial derivative of `1/c`. See [FULL_TRANSFORMATION.md](FULL_TRANSFORMATION.md) for the identity, interfaces, assumptions, and limitations.

## Implementations

- `solver.py`: one-dimensional periodic Fourier integration of the transformed equation and a Gaussian Duhamel estimator for a supplied source.
- `full_residual_nd.py`: direct multidimensional full residual ETD2 integration and an independent FD4/RK4 reference.
- `benchmark.py`: manufactured advection–diffusion refinement experiment.
- `plot_validation.py`: manufactured-solution figure and generated LaTeX numbers.
- `multidimensional_validation.py`, `validation_loop.py`, and `refine_failed_cases.py`: manufactured verification, game refinement, and sensitivity loops.
- `render_multidimensional_validation.py` and `render_population_validation.py`: figures from current native fields; producer font metadata is saved alongside each raster.
- `generate_royal_soc_animations.py`: matched GIF comparisons of LevelsetPy reference fields, full residual approximations, and absolute value errors for starling, double-integrator, Dubins, and rocket cases.
- `test_solver.py`: chain-rule, inverse-gradient, refinement, positivity, and Gaussian-integral checks.

Run from this directory:

```sh
python -m pip install -r requirements.txt
python -m pytest -q test_solver.py
python -m benchmark
python -m plot_validation --help
python -m validation_loop --stage refinement --workers 3
python -m validation_loop --stage sensitivity --workers 3
python -m validation_report --selection results/multidimensional/selection.json
```

Generate the Royal Society GIFs from the saved archives without rerunning the PDEs. The source
defaults to `results/multidimensional`; pass the blog animation directory explicitly:

```sh
python generate_royal_soc_animations.py \
  --output /path/to/robotsorcerer.github.io/downloads/Papers/assets/anim
```

Each frame uses matching saved times and solver settings. Its columns show the independent
LevelsetPy reference, the full residual approximation, and the absolute value difference. The
error color scale is shared across time within each GIF and reported in the final animation
manifest. Maximum errors appear in scientific notation in each frame heading, and colorbar ticks
use scientific notation as well. Rocket comparisons require matching the coefficient, horizon,
grid, and viscosity. The script checks these settings before rendering.

The `full` mode is the theoretical model. The `drop_derivative` and `heat` modes in the manufactured benchmark are negative controls that solve altered equations; their presence tests the necessity of the full residual. The quadratic constant-coefficient heat limit remains a valid special case.

## Evidence and scope

`full_transformation_benchmark_results.json` records manufactured-solution errors. Gaussian Duhamel sampling evaluates one supplied-source map; it does not implement an entire stochastic nonlinear source iteration. Periodic spatial grids require storage growing with dimension. Neither routine establishes high-dimensional efficiency, reach–avoid guarantees, or population safety certificates.

Current multidimensional runs solve the complete transformed equation directly, with a fixed collocation source-mode cutoff. Defects are checked against the untruncated equation. The original fixed-per-step-filter reference and inconsistent game implementations have been removed. See `results/multidimensional/VALIDATION_REPORT.md` and `LOOP_STATE.md` for the current numerical status; failed and pending checks remain explicit. A numerical pass does not establish editorial acceptance or a continuum certificate.

`population_validation.py` saves all 100,000 fresh query states, values, paired references, counts, seed, and archive hashes. It evaluates one altitude-invariant relative-state field; it does not simulate a joint flock trajectory.

`write_validation_tex.py` generates the manuscript tables and numeric macros from native results. Final figures resolve under the manuscript's `arxiv/figures/`. Re-rendering existing fields does not rerun the PDE. Source and data digests must match before using cached results.

The package exports only full residual interfaces. The audit and removed-path inventory are in [THEORY_AUDIT.md](THEORY_AUDIT.md) and `theory_audit_removed_paths.json`.
