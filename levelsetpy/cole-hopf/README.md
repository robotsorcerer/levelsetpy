# Cole--Hopf nested Picard scaling

This directory implements a depth-first nonlinear Monte Carlo Picard evaluator
for point queries of the transformed viscous Hamilton--Jacobi equation. It is
separate from the deterministic tensor-grid solvers under `monte_carlo/hjgauss`.

The evaluator retains the complete transformed source. Level zero evaluates
homogeneous heat flow. Each higher level freezes the nonlinear source at the
preceding Picard level and evaluates the Duhamel value and score-gradient
formulas by Gaussian sampling. Source-time importance sampling controls the
terminal-time gradient singularity.

Depth-first recursion avoids a state-space tensor grid. For fixed Picard depth,
the live coordinate payload is linear in sample count and state dimension.
Runtime is not dimension independent: nested work grows rapidly with sample
count and Picard depth. The benchmark reports both properties.

## Run

From this directory:

```sh
python -m pytest -q test_picard.py
python scaling.py --output results/scaling.json --replicates 3
python plot_scaling.py --input results/scaling.json --output results/scaling.pdf
```

`paper_assets.py` reads the same JSON record and generates the manuscript PNG,
companion scaling GIF, and LaTeX number macros. It does not rerun or modify the
benchmark measurements.

The JSON record contains source hashes, environment metadata, every replicate,
operation counts, gross and incremental peak resident memory, Python-traced
peak memory, and log--log slopes. The CSV file contains the rows used for plots.

## Claim scope

The results measure point-query evaluation for a bounded advection--diffusion
problem with a nonlinear Cole--Hopf source and a known viscous solution. They
can support a grid-free point-query storage claim. They do not establish
favorable end-to-end runtime, continuum field reconstruction, reachable-set
accuracy between queries, or a population-wide safety certificate.
