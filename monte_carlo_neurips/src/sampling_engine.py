__all__= [
    "SolverConfig",
    "cole_hopf_forward",
    "cole_hopf_inverse",
    "compute_frozen_coefficient",
    "relative_residual",
    "mc_value_at_point",
    "mc_gradient_at_point",
    "mc_value_batch",
    "mc_gradient_batch",
    "solve_quasi_linear",
    "sphere_cost",
    "cylinder_cost",
    "extract_zero_levelset_2d",
]

"""Shared utilities for the quasi-linearization HJ reachability solver.

Implements the core machinery of Algorithm 1 (Quasi-Linearized Cole-Hopf)
from the ICML 2026 paper:

  "Approximately Correct and Scalable HJ-Reachability: A Sampling Scheme"

Notation follows the paper:
  - v(t; x)  : value function (viscosity solution of HJ PDE)
  - v_t      : partial derivative of v w.r.t. time
  - Dv       : spatial gradient of v
  - omega    : Cole-Hopf transformed variable, omega = exp(-c * v)
  - c(t,x)   : frozen coefficient, c = 2 H^delta / (delta |Dv|^2)
  - delta     : viscosity parameter (> 0)
  - g(x)     : terminal / initial cost (signed distance to target set)
  - H(t,x,p) : Hamiltonian of the differential game

Key equations:
  - (Linear-Trans):  omega^delta := exp(-c * v^delta)
  - (Linear-HJ):     omega_t - (delta/2) Delta omega = 0
  - (Lemma 3.2):     v^delta(t;x) = -(1/c) log E_{y~N(x, delta*t)} [exp(-c g(y))]
  - (Eq 13/B.14):    Dv = (1/(t*delta*c)) (x - E[y exp(-cg(y))]/E[exp(-cg(y))])
  - (Eq 18):         v(t;x) = -(1/c) log (1/N) sum exp(-c g(x + sqrt(delta*(T-t)) y_i))

References to the paper are given as (Eq N) or (Lemma N.M) throughout.
"""

import numpy as np
from typing import Callable, Tuple, List, Optional, Dict
from dataclasses import dataclass, field


# ============================================================================
#  Configuration
# ============================================================================

@dataclass
class SolverConfig:
    """Configuration for the quasi-linearization solver.

    Attributes
    ----------
    delta : float
        Viscosity parameter. Controls smoothing of the HJ PDE.
        The approximation error is O(sqrt(delta)) by Crandall-Lions.
    num_samples : int
        Number of Monte Carlo samples (N in Eq 18).
    max_iters : int
        Maximum quasi-linearization iterations (Algorithm 1 loop bound).
    tol : float
        Convergence tolerance epsilon for ||v^(k+1) - v^(k)|| / ||v^(k)||.
    t_start : float
        Start time of the backward reachability computation.
    t_end : float
        Terminal time T (end of backward horizon).
    time_steps : int
        Number of discrete time steps for backward solve.
    seed : int
        Random seed for reproducibility.
    smoothing_eps : float
        Smoothing parameter for |.| approximation in Hamiltonians.
    """
    delta: float = 0.08
    num_samples: int = 10_000
    max_iters: int = 20
    tol: float = 1e-5
    t_start: float = 0.0
    t_end: float = 1.0
    time_steps: int = 50
    seed: int = 42
    smoothing_eps: float = 1e-4


# ============================================================================
#  Cole-Hopf Transformation Utilities (Pure NumPy)
# ============================================================================

def cole_hopf_forward(v: np.ndarray, c: np.ndarray) -> np.ndarray:
    r"""Forward Cole-Hopf transformation: omega = exp(-c * v).

    Implements (Linear-Trans) from the paper:
        omega^delta := exp(-c * v^delta)

    Parameters
    ----------
    v : np.ndarray
        Value function values.
    c : np.ndarray
        Frozen coefficient, broadcastable to v.

    Returns
    -------
    np.ndarray
        Transformed variable omega.
    """
    return np.exp(-c * v)


def cole_hopf_inverse(omega: np.ndarray, c: np.ndarray) -> np.ndarray:
    r"""Inverse Cole-Hopf: v = -(1/c) * log(omega).

    Implements Step 3 of Algorithm 1:
        v^{k+1} = -(1/c^{k}) * log(omega^{k})

    Parameters
    ----------
    omega : np.ndarray
        Transformed variable (must be > 0).
    c : np.ndarray
        Frozen coefficient (must be > 0).

    Returns
    -------
    np.ndarray
        Recovered value function.
    """
    omega_safe = np.maximum(omega, 1e-30)
    return -(1.0 / c) * np.log(omega_safe)


def compute_frozen_coefficient(
    H_vals: np.ndarray,
    grad_v_sq: np.ndarray,
    delta: float,
    eps: float = 1e-8,
) -> np.ndarray:
    r"""Compute the frozen coefficient c(t,x) = 2 H / (delta |Dv|^2).

    This is the key quantity from the quasi-linearization that connects
    the general Hamiltonian to the heat equation. See:
      - Appendix B, Eq (B.4): phi^delta = exp(-2/delta * H^delta / |Dz|^2 * z)
      - Algorithm 1, initialization: c^{(0)} = 2 H(t;x, Dg) / (delta |Dg|^2)
      - Algorithm 1, Step 4: c^{(k+1)} = 2 H(t;x, Dv^{(k+1)}) / (delta |Dv^{(k+1)}|^2)

    Parameters
    ----------
    H_vals : np.ndarray
        Hamiltonian values H(t, x, Dv), shape (M,).
    grad_v_sq : np.ndarray
        Squared norm of gradient |Dv|^2, shape (M,).
    delta : float
        Viscosity parameter.
    eps : float
        Regularization to prevent division by zero.

    Returns
    -------
    np.ndarray
        Frozen coefficient c, shape (M,).
    """
    return 2.0 * H_vals / (delta * np.maximum(grad_v_sq, eps))


def relative_residual(v_new: np.ndarray, v_old: np.ndarray, eps: float = 1e-10) -> float:
    r"""Relative L2 residual: ||v_new - v_old|| / ||v_old||.

    Algorithm 1, Step 5 convergence check.

    Parameters
    ----------
    v_new, v_old : np.ndarray
        Successive iterates of the value function.

    Returns
    -------
    float
        Relative residual.
    """
    norm_old = np.linalg.norm(v_old)
    if norm_old < eps:
        return np.linalg.norm(v_new - v_old)
    return float(np.linalg.norm(v_new - v_old) / norm_old)


# ============================================================================
#  Monte Carlo Value Function and Gradient (Pure NumPy)
# ============================================================================

def mc_value_at_point(
    x: np.ndarray,
    t: float,
    T: float,
    delta: float,
    c: float,
    g_fn: Callable[[np.ndarray], float],
    num_samples: int,
    rng: np.random.Generator,
) -> float:
    r"""Compute v^delta(t; x) via Monte Carlo Gaussian expectation.

    Implements Corollary 3.4 / Eq (18) with log-sum-exp for stability:

        v^delta(t; x) = -(1/c) log (1/N) sum_{i=1}^{N}
                         exp(-c * g(x + sqrt(delta*(T-t)) * y_i))

    where y_i ~ N(x, \delta * t * I_n).

    For BRT (backward reachable tube), g(y) transforms to
    g(x + sqrt(delta*(T-t)) * y) per Corollary B.2.

    Parameters
    ----------
    x : np.ndarray, shape (n,)
        Query point in state space.
    t : float
        Current time.
    T : float
        Terminal time.
    delta : float
        Viscosity parameter.
    c : float
        Cole-Hopf coefficient (must be > 0).
    g_fn : callable
        Terminal cost function g: R^n -> R.
    num_samples : int
        Number of MC samples N.
    rng : np.random.Generator
        NumPy random number generator.

    Returns
    -------
    float
        MC estimate of v(t; x).
    """
    n = x.shape[0]
    sigma = np.sqrt(max(delta * (T - t), 1e-30))

    # Draw y_i ~ N(x, delta*(T-t)*I) = x + sigma * z_i, z_i ~ N(0,I)
    z = rng.standard_normal((num_samples, n))
    y = x[np.newaxis, :] + sigma * z  # (N, n)

    # Evaluate terminal cost at samples
    g_vals = np.array([g_fn(yi) for yi in y])  # (N,)

    # Log-sum-exp for numerical stability (Eq 18)
    exponents = -c * g_vals
    max_exp = np.max(exponents)
    log_mean_exp = max_exp + np.log(np.mean(np.exp(exponents - max_exp)))

    return -(1.0 / c) * log_mean_exp


def mc_gradient_at_point(
    x: np.ndarray,
    t: float,
    T: float,
    delta: float,
    c: float,
    g_fn: Callable[[np.ndarray], float],
    num_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    r"""Compute Dv^delta(t; x) via importance-weighted MC.

    Implements Corollary 3.5 / Eq (19):

        Dv^delta(t; x) = (1/(t*delta*c)) * (x - sum w_i d_i / sum w_i)

    where d_i = x + sqrt(delta*(T-t)) * y_i, w_i = exp(-c * g(d_i)),
    and y_i ~ N(x, \delta(T-t)y_i).

    Equivalently (Eq 13 / B.14):
        Dv = (1/(t*delta*c)) (x - E[y exp(-cg(y))] / E[exp(-cg(y))])

    Parameters
    ----------
    (same as mc_value_at_point)

    Returns
    -------
    np.ndarray, shape (n,)
        Gradient estimate Dv(t; x).
    """
    n = x.shape[0]
    sigma = np.sqrt(max(delta * (T - t), 1e-30))
    t_eff = max(T - t, 1e-30)  # effective time for gradient formula

    z = rng.standard_normal((num_samples, n))
    d = x[np.newaxis, :] + sigma * z  # (N, n) -- sample points

    g_vals = np.array([g_fn(di) for di in d])  # (N,)

    # Importance weights (stabilized)
    log_w = -c * g_vals
    log_w_shifted = log_w - np.max(log_w)
    weights = np.exp(log_w_shifted)
    weights = weights / np.sum(weights)  # normalized

    # Weighted mean of sample points
    weighted_mean = np.sum(weights[:, np.newaxis] * d, axis=0)  # (n,)

    # Gradient from Eq (19) / (B.14)
    Dv = (1.0 / (t_eff * delta * c)) * (x - weighted_mean)
    return Dv


def mc_value_batch(
    eval_points: np.ndarray,
    t: float,
    T: float,
    delta: float,
    c: np.ndarray,
    g_fn: Callable[[np.ndarray], float],
    num_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Batch evaluation of v(t; x) at M points.

    Parameters
    ----------
    eval_points : np.ndarray, shape (M, n)
        Query points.
    c : np.ndarray, scalar or shape (M,)
        Cole-Hopf coefficient(s).

    Returns
    -------
    np.ndarray, shape (M,)
        Value function estimates.
    """
    M = eval_points.shape[0]
    c_arr = np.broadcast_to(np.asarray(c), (M,))
    results = np.zeros(M)
    for i in range(M):
        results[i] = mc_value_at_point(
            eval_points[i], t, T, delta, c_arr[i],
            g_fn, num_samples, rng,
        )
    return results


def mc_gradient_batch(
    eval_points: np.ndarray,
    t: float,
    T: float,
    delta: float,
    c: np.ndarray,
    g_fn: Callable[[np.ndarray], float],
    num_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Batch evaluation of Dv(t; x) at M points.

    Returns
    -------
    np.ndarray, shape (M, n)
        Gradient estimates.
    """
    M, n = eval_points.shape
    c_arr = np.broadcast_to(np.asarray(c), (M,))
    results = np.zeros((M, n))
    for i in range(M):
        results[i] = mc_gradient_at_point(
            eval_points[i], t, T, delta, c_arr[i],
            g_fn, num_samples, rng,
        )
    return results


# ============================================================================
#  Quasi-Linearization Solver (Algorithm 1) -- Pure NumPy
# ============================================================================

def solve_quasi_linear(
    eval_points: np.ndarray,
    t: float,
    T: float,
    delta: float,
    c_init: np.ndarray,
    g_fn: Callable[[np.ndarray], float],
    H_fn: Callable[[float, np.ndarray, np.ndarray], float],
    num_samples: int,
    max_iters: int,
    tol: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, List[float]]:
    r"""Quasi-linearization iteration (Algorithm 1) for general H.

    Algorithm 1: Quasi-Linearized Cole-Hopf
    ----------------------------------------
    Require: v^{(0)}(t;x) = g(x),  c^{(0)} = 2 H(t;x, Dg) / (delta |Dg|^2)

    For k = 0, 1, 2, ...:
      1. Freeze c^{(k)} at the current iterate.
      2. Solve heat equation omega_t = (delta/2) Delta omega
         with initial data omega^{(k)}(0; x) = exp(-c^{(k)} * g(x)).
         [Done via MC Gaussian expectation, Eq (18)]
      3. Recover v^{(k+1)} = -(1/c^{(k)}) * log(omega^{(k)}).
      4. Update Dv^{(k+1)} [via Eq (19)] and
         c^{(k+1)} = 2 H(t;x, Dv^{(k+1)}) / (delta |Dv^{(k+1)}|^2).
      5. Check convergence: ||v^{(k+1)} - v^{(k)}|| / ||v^{(k)}|| < epsilon.

    Parameters
    ----------
    eval_points : np.ndarray, shape (M, n)
        Query points in state space.
    t : float
        Current time.
    T : float
        Terminal time.
    delta : float
        Viscosity parameter.
    c_init : np.ndarray, shape (M,)
        Initial frozen coefficient c^{(0)} for each point.
    g_fn : callable
        Terminal cost g: R^n -> R.
    H_fn : callable
        Hamiltonian H(t, x, p) -> scalar.
    num_samples : int
        MC sample count.
    max_iters : int
        Maximum iterations.
    tol : float
        Convergence tolerance.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    v : np.ndarray, shape (M,)
        Converged value function estimate.
    history : list of float
        Relative residual at each iteration.
    """
    M = eval_points.shape[0]

    # Initialize: v^{(0)}(t;x) = g(x)
    v_current = np.array([g_fn(xi) for xi in eval_points])
    c_frozen = np.abs(c_init) + 1e-8  # ensure c > 0

    history: List[float] = []

    for k in range(max_iters):
        # Step 1: c^{(k)} is frozen from previous iteration (or initialization)

        # Step 2 & 3: Solve heat equation and recover v^{(k+1)}
        # The MC expectation (Eq 18) directly gives v, combining steps 2-3
        v_new = mc_value_batch(
            eval_points, t, T, delta, c_frozen,
            g_fn, num_samples, rng,
        )

        # Step 5: Check convergence
        resid = relative_residual(v_new, v_current)
        history.append(resid)
        v_current = v_new
        if k%4 == 0 or k == max_iters-1:
            print(f"Iteration {k+1}: residual: {resid:.6f}, max v = {np.max(v_new):.4f}, min v = {np.min(v_new):.4f}")

        if resid < tol:
            break

        # Step 4: Update gradient and frozen coefficient
        Dv = mc_gradient_batch(
            eval_points, t, T, delta, c_frozen,
            g_fn, num_samples, rng,
        )

        # Evaluate Hamiltonian at (t, x, Dv)
        H_vals = np.array([
            H_fn(t, eval_points[i], Dv[i]) for i in range(M)
        ])

        # c^{(k+1)} = 2 H(t; x, Dv^{(k+1)}) / (delta |Dv^{(k+1)}|^2)
        grad_v_sq = np.sum(Dv ** 2, axis=-1)
        c_frozen = compute_frozen_coefficient(H_vals, grad_v_sq, delta)
        c_frozen = np.abs(c_frozen) + 1e-8  # ensure positivity

    return v_current, history


# ============================================================================
#  Terminal cost / target set functions
# ============================================================================

def sphere_cost(x: np.ndarray, center: Optional[np.ndarray] = None,
                radius: float = 1.0) -> float:
    """Signed distance to a sphere: ||x - center|| - radius.

    Negative inside the target set, positive outside.
    """
    if center is None:
        center = np.zeros_like(x)
    return float(np.linalg.norm(x - center) - radius)


def cylinder_cost(x: np.ndarray, axis_align: int = 2,
                  center: Optional[np.ndarray] = None,
                  radius: float = 1.5) -> float:
    """Signed distance to an axis-aligned cylinder.

    The cylinder is infinite along dimension ``axis_align``.
    For the two-rockets problem, axis_align=2 (theta dimension).
    """
    if center is None:
        center = np.zeros_like(x)
    diff = x - center
    mask = np.ones(x.shape[-1])
    mask[axis_align] = 0.0
    return float(np.linalg.norm(diff * mask) - radius)


# ============================================================================
#  Visualization helpers
# ============================================================================

def extract_zero_levelset_2d(
    X: np.ndarray, Y: np.ndarray, V: np.ndarray
) -> Optional[list]:
    """Extract zero-level contour from a 2D value function grid.

    Returns matplotlib contour paths or None if no contour found.
    """
    import matplotlib.pyplot as plt
    fig_tmp, ax_tmp = plt.subplots()
    cs = ax_tmp.contour(X, Y, V, levels=[0.0])
    paths = cs.collections[0].get_paths() if cs.collections else []
    plt.close(fig_tmp)
    return paths if paths else None
