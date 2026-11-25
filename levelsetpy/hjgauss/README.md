# Sampling-based HJ Reachability (HJ-Gauss)

Paper: [Sampling-based HJ Reachability](https://scriptedonachip.com/downloads/Papers/roysoc.pdf)

## Objective

This project implements the paper's grid-free point-query algorithm for the transformed viscous Hamilton--Jacobi equation. The evaluator:

1. applies the generalized Cole--Hopf map;
2. treats level zero as homogeneous heat flow;
3. freezes the complete nonlinear source at the preceding Picard level;
4. estimates the Duhamel value and score-gradient formulas with antithetic Gaussian samples; and
5. uses source-time importance sampling to control the terminal-time gradient singularity.

Recursion is depth first, so the implementation does not allocate a state-space tensor grid. At fixed Picard depth, its live coordinate payload is linear in sample count and state dimension. The nested tree of preceding-level queries means runtime grows rapidly with both sample count and depth.

The deterministic transform, field solvers, independent references, and multidimensional validation used by the paper live in [`../cole-hopf/`](../cole-hopf/). They are shared evidence, not a second copy of the Monte Carlo implementation.

## Layout

| Path | Purpose |
|---|---|
| `picard.py` | Depth-first nonlinear Monte Carlo Picard point evaluator |
| `scaling.py` | Fresh-process dimension, sample-count, and Picard-depth sweeps |
| `plot_scaling.py` | Plot the native JSON record without recomputing it |
| `paper_assets.py` | Generate the manuscript figure, animation, and LaTeX macros from the same record |
| `tests/test_picard.py` | Reproducibility, positivity, work-count, storage, and input checks |
| `results/` | Checked-in native scaling record, report, CSV, and rendered plots |

## Recorded results

The checked-in benchmark uses a bounded advection--diffusion HJ problem with a known viscous solution. Each row is the median of three fresh-process runs.

| Sweep | Fixed settings | Range | Median runtime |
|---|---|---|---:|
| State dimension | 12 samples, Picard level 2 | 8 to 2048 | `0.0437 s` to `0.0748 s` |
| Samples per update | dimension 8, Picard level 2 | 4 to 48 | `0.0050 s` to `0.9103 s` |
| Picard level | dimension 8, 6 samples | 0 to 3 | `0.00059 s` to `0.05296 s` |

Across the dimension sweep, Python-traced peak memory increases from `0.028 MiB` to `1.180 MiB`, while the directly counted depth-linear coordinate payload increases from `0.004 MiB` to `0.938 MiB`. The fitted log--log runtime slope versus dimension is `0.082`; the sample-count runtime slope is `2.06`. Absolute value errors in the dimension sweep remain below `5.6e-4`.

These measurements support a grid-free point-query storage claim for fixed Picard depth. They do not establish dimension-independent runtime, favorable scaling in Picard depth, continuum field reconstruction, reachable-set topology between queries, or a population-wide safety certificate.

See [`results/SCALING_REPORT.md`](results/SCALING_REPORT.md) for the concise report and `results/scaling.json` for replicate records, operation counts, source hashes, and environment metadata.

## Install and test

Run from this directory:

```sh
python -m pip install -e '.[dev]'
python -m pytest -q
```

## Reproduce the scaling study

```sh
python scaling.py --output results/scaling.json --replicates 3
python plot_scaling.py \
  --input results/scaling.json \
  --output results/scaling.pdf
```

`paper_assets.py` reads the same JSON record and does not rerun or modify the benchmark:

```sh
python paper_assets.py \
  --input results/scaling.json \
  --figure /path/to/colehopf_scaling.png \
  --gif /path/to/roysoc_colehopf_scaling.gif \
  --macros /path/to/colehopf_scaling_numbers.tex
```

Re-running `scaling.py` replaces the native timing record with measurements from the current machine and environment.
