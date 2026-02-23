"""Error metrics, convergence analysis, and residual computation."""

from typing import Dict

import jax.numpy as jnp


def compute_error_metrics(
    v_sample: jnp.ndarray, v_ref: jnp.ndarray
) -> Dict[str, float]:
    """Compare sampling-based solution to a reference (e.g. grid solver).

    Parameters
    ----------
    v_sample : (M,) sampled values.
    v_ref    : (M,) reference values.

    Returns
    -------
    dict with l2_relative, l_inf, rmse, mean_abs_error.
    """
    diff = v_sample - v_ref
    ref_norm = jnp.maximum(jnp.linalg.norm(v_ref), 1e-12)
    return {
        "l2_relative": float(jnp.linalg.norm(diff) / ref_norm),
        "l_inf": float(jnp.max(jnp.abs(diff))),
        "rmse": float(jnp.sqrt(jnp.mean(diff ** 2))),
        "mean_abs_error": float(jnp.mean(jnp.abs(diff))),
    }


def convergence_rate(
    errors: jnp.ndarray, sample_counts: jnp.ndarray
) -> float:
    """Estimate MC convergence rate: error ~ C * N^(-alpha).

    Returns alpha (ideally ~0.5 for standard MC).
    """
    log_N = jnp.log(jnp.asarray(sample_counts, dtype=jnp.float32))
    log_err = jnp.log(jnp.asarray(errors, dtype=jnp.float32))
    coeffs = jnp.polyfit(log_N, log_err, 1)
    return -float(coeffs[0])
