# Full generalized Cole–Hopf transformation

This note documents source files stored at the root of `monte_carlo/`.

The NumPy solvers retain both algebraic and coefficient-derivative residuals. They prescribe a smooth positive coefficient.

The equation uses **elapsed time from the initial condition**: `v_t + H(t,x,v_x) = delta*v_xx/2`. Choose a prescribed smooth positive coefficient `c(t,x)` and set `omega=exp(-c*v)`. The code evaluates the full chain-rule source `omega*(R_alg+R_der)` from `omega, omega_x, c, c_t, c_x, c_xx`. Inversion includes the derivative of `1/c`. A prescribed coefficient avoids divisions by `|Dv|²`, including at critical points, while retaining the original Hamiltonian through `R_alg`.

## Run

From the containing `monte_carlo` directory :

```sh
python -m pytest -q test_solver.py
python -m benchmark
```

Dependencies are NumPy and pytest. The benchmark writes
`full_transformation_benchmark_results.json`.

## Interfaces

- `residual`: pure algebraic and derivative residual evaluation in one space dimension.
- `inverse_gradient`: full inverse spatial derivative.
- `solve_periodic`: Fourier collocation with first-order exponential Euler time stepping on a periodic one-dimensional domain. A coefficient callback supplies a `Coefficient` containing all four fields. A Hamiltonian callback takes `(t,x,p)`.
- `gaussian_duhamel`: real-line Monte Carlo evaluation of one linear Duhamel map with supplied initial condition and frozen source. It returns the transformed value, its spatial gradient, and empirical standard errors. These standard errors are diagnostics, not confidence certificates. The time proposal concentrates samples near the singular endpoint in the gradient kernel.

Comparison modes `drop_derivative` and `heat` deliberately solve altered equations. Full-source integration can use an arbitrary positive coefficient; algebraic matching is neither imposed nor needed here.

## Evidence and limits

The manufactured advection-diffusion solution is `v(t,x)=exp(-delta*t/2)*sin(x-.7*t)`, using `delta=.4`, horizon `.5`, periodic length `2*pi`, and `c=1+.2*cos(x)+.1*sin(t)`. The tests check the chain-rule identity, the inverse gradient, the quadratic heat limit, temporal refinement, retained-source versus discarded-source behavior, guards, and a known Gaussian Duhamel integral. The benchmark records deterministic errors against the manufactured solution; no stochastic error is involved in the Fourier benchmark.

The Fourier prototype stores a spatial grid, is limited to one periodic space dimension, and does not implement obstacles, reach-avoid stopping, adaptive coefficient matching, neural/source reconstruction, high-dimensional scaling, or a certified uniform error bound. Products are not dealiased. Positivity failures raise an exception rather than clamp the transformed solution; explicit source integration requires a suitable time step. This is a diagnostic implementation of the retained-source equation, not a replacement production reachability solver.

The Gaussian Duhamel helper requires source values at arbitrary space-time samples. It does not solve that reconstruction problem and is not a recursive Monte Carlo algorithm. Fresh independent draws and a fixed supplied source make its sample averages unbiased when the indicated moments exist. Logarithms and normalized gradient ratios would introduce finite-sample bias if applied afterward.

## Manufactured-solution validation figure

`python -m plot_validation --out-dir PATH` recomputes the manufactured records
and generated LaTeX numbers before rendering. Use `--figure-dir PATH` to set a
separate PNG destination. The figure compares full-source and homogeneous-heat
series; derivative-omission data remain in the native records and tests but are
not plotted. This manufactured-solution figure is separate from the exploratory
multidimensional game diagnostics described in `README.md`.

## Multidimensional validation

`full_residual_nd.py` integrates the complete transformed equation with ETD2 and a fixed collocation source cutoff. A separate FD4/RK4 implementation evolves the original equation at matched viscosity. The untruncated sampled defect exposes the cutoff error. Exact manufactured solutions in two and three dimensions test both solvers. `multidimensional_validation.py` defines the physical controls, elapsed-time Hamiltonians, bounded initial conditions, and periodic completion. The reproducibility loop uses fixed acceptance thresholds; its live status is recorded in `results/multidimensional/validation_summary.json`.

The fixed source cutoff is a numerical approximation, not omission of an analytic residual. The time-dependent coefficient is smooth, positive, and prescribed. All coefficient derivatives and the full inverse gradient are retained. Negative-control source omission remains confined to the manufactured tests.
