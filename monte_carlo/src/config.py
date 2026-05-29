"""Solver configuration as plain NamedTuples (JAX PyTree-compatible)."""

from typing import NamedTuple


class SolverConfig(NamedTuple):
    """Immutable solver configuration.

    All fields are scalars or tuples so this is a valid JAX PyTree leaf
    when registered, or can be passed as static_argnums.
    """
    delta: float = 0.1
    num_samples: int = 1_000
    max_quasi_iters: int = 20
    quasi_tol: float = 1e-6
    time_steps: int = 50
    t_start: float = 0.0
    t_end: float = 1.0
    seed: int = 123
    smoothing_eps: float = 1e-4
    gradient_mode: str = "b17"
    chunk_size: int = 5_000
    n_flocks: int = 1
    n_predators: int = 1
