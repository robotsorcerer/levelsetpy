"""Shared fixtures for the test suite."""

import pytest
import jax
import jax.numpy as jnp

from src.config import SolverConfig


@pytest.fixture
def rng():
    """Deterministic PRNG key."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def default_config():
    """Default solver configuration for tests."""
    return SolverConfig(
        delta=0.1,
        num_samples=5_000,
        max_quasi_iters=10,
        quasi_tol=1e-6,
        time_steps=10,
        t_start=0.0,
        t_end=1.0,
        seed=42,
        smoothing_eps=1e-4,
    )


@pytest.fixture
def tight_config():
    """High-sample-count config for convergence tests."""
    return SolverConfig(
        delta=0.1,
        num_samples=100_000,
        max_quasi_iters=20,
        quasi_tol=1e-8,
        time_steps=20,
        t_start=0.0,
        t_end=1.0,
        seed=42,
        smoothing_eps=1e-6,
    )
