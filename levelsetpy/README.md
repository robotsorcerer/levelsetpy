This package directory contains the grid-based solver modules and the research subprojects [`monte_carlo/`](monte_carlo/) and [`mapf/`](mapf/). Each subproject has its own README with usage, data provenance, and limitations. The Monte Carlo scripts are run from their directory root; the MAPF experiments are run from `mapf/`.

Monte Carlo code retains the full residual transformation only. Its game computations are exploratory; see `monte_carlo/THEORY_AUDIT.md` for scope and removed implementations.
