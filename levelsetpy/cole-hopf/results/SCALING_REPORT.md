# Cole--Hopf scaling results

The benchmark evaluates one transformed-PDE point with the depth-first
nonlinear Monte Carlo Picard implementation in `picard.py`.  Each entry is the
median of three fresh-process runs.  The test problem is a bounded
advection--diffusion Hamilton--Jacobi equation with an analytic viscous
solution; the Cole--Hopf source remains nonlinear.

## State dimension

The Picard level is 2 and the sample count is 12 per update.

| dimension | runtime (s) | traced peak (MiB) | coordinate payload (MiB) | absolute error |
|---:|---:|---:|---:|---:|
| 8 | 0.0437 | 0.028 | 0.004 | 5.52e-4 |
| 16 | 0.0443 | 0.033 | 0.007 | 5.53e-4 |
| 32 | 0.0449 | 0.042 | 0.015 | 3.78e-4 |
| 64 | 0.0453 | 0.061 | 0.029 | 1.06e-4 |
| 128 | 0.0463 | 0.098 | 0.059 | 6.00e-5 |
| 256 | 0.0484 | 0.172 | 0.117 | 1.31e-4 |
| 512 | 0.0531 | 0.320 | 0.234 | 5.10e-5 |
| 1024 | 0.0598 | 0.617 | 0.469 | 5.80e-5 |
| 2048 | 0.0760 | 1.180 | 0.938 | 6.00e-6 |

The fitted log--log runtime slope is 0.085 over this range.  The measured
Python-traced-memory slope is 0.694 because fixed interpreter overhead
dominates at small dimensions.  The directly counted live coordinate payload
is linear in dimension for fixed sample count and Picard depth.

## Sampling and nesting

At dimension 8 and Picard level 2, increasing samples per update from 4 to 48
increased median runtime from 0.0052 s to 0.9355 s.  The fitted log--log slope
is 2.05.  With six samples per update, increasing the Picard level from 0 to 3
increased runtime from 0.00062 s to 0.05399 s.

## Supported claim

These results support a **grid-free point-query storage** statement: the
implementation does not allocate a state-space tensor grid, and its live
coordinate payload is linear in state dimension and samples for fixed Picard
depth.  They do not support an `O(N n)` end-to-end runtime claim.  Nested
sampling grows rapidly with sample count and depth, and these tests do not
establish continuum reconstruction accuracy, reachable-set topology, or a
population-wide safety certificate.

Exact replicate data, operation counts, source hashes, and environment details
are in `scaling.json`; plot-ready rows are in `scaling.csv`.
