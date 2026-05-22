#!/usr/bin/env python
"""Two-Rockets Pursuit-Evasion: Pure Python/NumPy Implementation.

Implements the quasi-linearization algorithm (Algorithm 1) from:

    "Approximately Correct and Scalable HJ-Reachability: A Sampling Scheme"
    ICML 2026

for the rocket launch problem described in Section 4.1.

Problem Setup (Section 4.1, Eqs 20-24)
---------------------------------------
Two identical rockets P (pursuer) and E (evader) on an (x, z) plane.
Relative coordinates: state x = (x, z, theta) where
  - (x, z) in R^2: relative position of E w.r.t. P
  - theta in [-pi/2, pi/2]: relative thrust inclination (u_p - u_e)

Dynamics (Eq 20):
    x_dot     = a_p cos(theta) + u_e * x
    z_dot     = a_p sin(theta) + a_e + u_e * x - g
    theta_dot = u_p - u_e

where a = a_p = a_e = 1 ft/sec^2 and g = 32 ft/sec^2.

Hamiltonian (Eq 22-23): Setting a_e = a_p = a, u_e_bar = u_p_bar = u_bar:
    H(t; x, p) = -a p1 cos(theta) - p2(g - a - a sin(theta))
                 + u_bar |p2*x + p3| - u_bar |p1*x + p3|

Terminal condition: Capture region Phi = {||PE||_2 <= ell}
    with ell = 1.5 ft, so g(x) = sqrt(x^2 + z^2) - 1.5.
    The target set is a cylinder in (x, z, theta) space, infinite along theta.

Discretization (Eq 25):
    (x, z) in (-100, +100), theta in (-pi/2, pi/2), t in (0, 1]
"""

import numpy as np
from math import pi
from typing import Tuple, List, Optional, Callable
from functools import partial

from .config_loaders import _load_rocket_config

from src.sampling_engine import (
    SolverConfig,
    cole_hopf_forward,
    cole_hopf_inverse,
    compute_frozen_coefficient,
    relative_residual,
    mc_value_at_point,
    mc_gradient_at_point,
    mc_value_batch,
    mc_gradient_batch,
    solve_quasi_linear,
    cylinder_cost,
)



# Load config as NamedTuple
CONFIG = _load_rocket_config()


# ============================================================================
#  Hamiltonian (Eq 22-23)
# ============================================================================

class RocketsHamiltonian:
    """Two-rockets pursuit-evasion Hamiltonian in relative coordinates.

    From Eq (22)-(23) of the paper:

        H(t; x, p) = -max_{u_e} min_{u_p} [p1, p2, p3] . f(x, u_p, u_e)

    Setting a_e = a_p = a, u_e_lower = u_p_lower = -u_bar,
    u_e_upper = u_p_upper = +u_bar, the Hamiltonian simplifies to (Eq 23):

        H = -a p1 cos(theta) - p2(g - a - a sin(theta))
            + u_bar |p2*x + p3| - u_bar |p1*x + p3|

    Parameters
    ----------
    a : float
        Thrust magnitude (equal for both rockets).
    g : float
        Gravitational acceleration.
    u_bound : float
        Control bound (symmetric: u in [-u_bound, +u_bound]).
    smoothing_eps : float
        Smoothing for |z| ~ sqrt(z^2 + eps^2).
    """

    def __init__(
        self,
        a: float = CONFIG.A_THRUST,
        g: float = CONFIG.GRAVITY,
        u_bound: float = CONFIG.U_BOUND,
        smoothing_eps: float = 1e-4,
    ):
        self.a = a
        self.g = g
        self.u_bound = u_bound
        self.eps = smoothing_eps

    def __call__(self, t: float, x: np.ndarray, p: np.ndarray) -> float:
        """Evaluate H(t, x, p).

        Parameters
        ----------
        t : float
            Time (not used for time-invariant dynamics).
        x : np.ndarray, shape (3,)
            State [x_pos, z_pos, theta].
        p : np.ndarray, shape (3,)
            Co-state (spatial gradient Dv).

        Returns
        -------
        float
            Hamiltonian value.
        """
        p1, p2, p3 = p[0], p[1], p[2]
        x_pos = x[0]
        theta = x[2]

        smooth_abs = lambda z: np.sqrt(z ** 2 + self.eps ** 2)

        # Eq (23):
        # H = -a*p1*cos(theta) - p2*(g - a - a*sin(theta))
        #     + u_bar*|p2*x + p3| - u_bar*|p1*x + p3|
        term1 = -self.a * p1 * np.cos(theta)
        term2 = -p2 * (self.g - self.a - self.a * np.sin(theta))
        term3 = self.u_bound * smooth_abs(p2 * x_pos + p3)
        term4 = -self.u_bound * smooth_abs(p1 * x_pos + p3)

        return float(term1 + term2 + term3 + term4)

    @property
    def state_dim(self) -> int:
        return 3


# ============================================================================
#  Terminal cost function for the rockets problem
# ============================================================================

def rockets_terminal_cost(x: np.ndarray, radius: float = CONFIG.CAPTURE_RADIUS) -> float:
    r"""Terminal cost g(x) = sqrt(x^2 + z^2) - ell.

    This is the signed distance to the capture circle Phi = {||PE||_2 <= ell}.
    The cylinder_cost ignores the theta dimension (axis_align=2).

    Parameters
    ----------
    x : np.ndarray, shape (3,)
        State [x_pos, z_pos, theta].
    radius : float
        Capture radius ell.

    Returns
    -------
    float
        Signed distance (negative inside target, positive outside).
    """
    return cylinder_cost(x, axis_align=2, radius=radius)


# ============================================================================
#  Full solver for the rockets problem
# ============================================================================

class RocketsSolver:
    """Complete solver for the two-rockets pursuit-evasion problem.

    Wraps the quasi-linearization iteration (Algorithm 1) with the
    rockets-specific Hamiltonian and terminal cost.

    Parameters
    ----------
    config : SolverConfig
        Solver configuration.
    hamiltonian : RocketsHamiltonian, optional
        Custom Hamiltonian (defaults to standard rockets).
    """

    def __init__(
        self,
        config: Optional[SolverConfig] = None,
        hamiltonian: Optional[RocketsHamiltonian] = None,
    ):
        self.cfg = config or SolverConfig(
            delta=0.08,
            num_samples=10_000,
            max_iters=20,
            tol=1e-5,
            t_start=0.0,
            t_end=1.0,
            seed=123,
        )
        self.H = hamiltonian or RocketsHamiltonian()
        self.g_fn = rockets_terminal_cost
        self.rng = np.random.default_rng(self.cfg.seed)

    def _compute_initial_c(
        self, eval_points: np.ndarray, t: float,
    ) -> np.ndarray:
        """Compute initial c^{(0)} = 2 H(t;x,Dg) / (delta |Dg|^2).

        As specified in the Require line of Algorithm 1.
        """
        M = eval_points.shape[0]
        c_init = np.zeros(M)

        for i in range(M):
            xi = eval_points[i]

            # Dg: gradient of terminal cost at xi
            # For cylinder cost g(x) = ||x_{spatial}|| - R,
            # Dg = [x_pos / ||x_s||, z_pos / ||x_s||, 0]
            x_pos, z_pos = xi[0], xi[1]
            r = np.sqrt(x_pos ** 2 + z_pos ** 2)
            r_safe = max(r, 1e-10)
            Dg = np.array([x_pos / r_safe, z_pos / r_safe, 0.0])

            Dg_sq = np.dot(Dg, Dg)
            H_val = self.H(t, xi, Dg)

            if Dg_sq > 1e-8:
                c_init[i] = 2.0 * H_val / (self.cfg.delta * Dg_sq)
            else:
                c_init[i] = 1.0 / self.cfg.delta

        return c_init

    def solve_slice(
        self,
        theta_val: float,
        grid_res: int = 35,
        domain: Tuple[float, float] = (-5.0, 5.0),
        t_eval: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[float]]:
        """Solve on a 2D (x, z) slice at fixed theta.

        Parameters
        ----------
        theta_val : float
            Fixed theta value for the slice.
        grid_res : int
            Number of grid points per spatial axis.
        domain : tuple
            (min, max) for x and z coordinates.
        t_eval : float
            Time at which to evaluate (typically 0 for BRT at final time).

        Returns
        -------
        X, Z : np.ndarray, shape (grid_res, grid_res)
            Meshgrid coordinates.
        V : np.ndarray, shape (grid_res, grid_res)
            Value function on the grid.
        history : list of float
            Convergence history.
        """
        xs = np.linspace(domain[0], domain[1], grid_res)
        X, Z = np.meshgrid(xs, xs, indexing="ij")

        theta_col = np.full((grid_res * grid_res, 1), theta_val)
        eval_points = np.concatenate(
            [X.ravel()[:, np.newaxis], Z.ravel()[:, np.newaxis], theta_col],
            axis=-1,
        )

        c_init = self._compute_initial_c(eval_points, t_eval)

        v, history = solve_quasi_linear(
            eval_points=eval_points,
            t=t_eval,
            T=self.cfg.t_end,
            delta=self.cfg.delta,
            c_init=c_init,
            g_fn=self.g_fn,
            H_fn=self.H,
            num_samples=self.cfg.num_samples,
            max_iters=self.cfg.max_iters,
            tol=self.cfg.tol,
            rng=self.rng,
        )

        V = v.reshape(grid_res, grid_res)
        return X, Z, V, history

    def solve_volume(
        self,
        grid_res: int = 25,
        spatial_domain: Tuple[float, float] = (-5.0, 5.0),
        theta_range: Tuple[float, float] = (-pi / 2, pi / 2),
        t_eval: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[float]]:
        """Solve on the full 3D (x, z, theta) grid.

        Returns
        -------
        XX, ZZ, TT : np.ndarray, shape (grid_res, grid_res, grid_res)
            Meshgrid coordinates.
        V : np.ndarray, same shape
            3D value function.
        history : list of float
            Convergence history.
        """
        xs = np.linspace(spatial_domain[0], spatial_domain[1], grid_res)
        ts = np.linspace(theta_range[0], theta_range[1], grid_res)
        XX, ZZ, TT = np.meshgrid(xs, xs, ts, indexing="ij")

        eval_points = np.stack(
            [XX.ravel(), ZZ.ravel(), TT.ravel()], axis=-1,
        )

        c_init = self._compute_initial_c(eval_points, t_eval)

        v, history = solve_quasi_linear(
            eval_points=eval_points,
            t=t_eval,
            T=self.cfg.t_end,
            delta=self.cfg.delta,
            c_init=c_init,
            g_fn=self.g_fn,
            H_fn=self.H,
            num_samples=self.cfg.num_samples,
            max_iters=self.cfg.max_iters,
            tol=self.cfg.tol,
            rng=self.rng,
        )

        V = v.reshape(grid_res, grid_res, grid_res)
        return XX, ZZ, TT, V, history


# ============================================================================
#  Plotting
# ============================================================================

def plot_2d_slices(
    solver: RocketsSolver,
    theta_slices: Optional[List[float]] = None,
    grid_res: int = 35,
    domain: Tuple[float, float] = (-5.0, 5.0),
    save_path: Optional[str] = None,
) -> None:
    """Plot 2D (x, z) slices of the BRT at several theta values.

    Parameters
    ----------
    solver : RocketsSolver
        Configured solver instance.
    theta_slices : list of float, optional
        Theta values for cross-sections. Default: [-pi/4, 0, pi/4, pi/2].
    grid_res : int
        Grid resolution per axis.
    domain : tuple
        Spatial domain (min, max).
    save_path : str, optional
        Path to save the figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if theta_slices is None:
        theta_slices = [-pi / 4, 0.0, pi / 4, pi / 2]

    n_slices = len(theta_slices)
    fig, axes = plt.subplots(
        1, n_slices, figsize=(4 * n_slices, 4),
        sharex=True, sharey=True,
    )
    if n_slices == 1:
        axes = [axes]

    for ax, theta_val in zip(axes, theta_slices):
        print(f"  Solving theta = {theta_val:.3f} ...")
        X, Z, V, history = solver.solve_slice(
            theta_val, grid_res=grid_res, domain=domain,
        )

        ax.contourf(X, Z, V, levels=20, cmap="RdBu_r")
        ax.contour(X, Z, V, levels=[0.0], colors="k", linewidths=2)
        ax.set_title(
            rf"$\theta = {theta_val:.2f}$  ({len(history)} iters)",
            fontsize=11, fontweight="bold",
        )
        ax.set_xlabel(r"$x$ (ft)", fontsize=11, fontweight="bold")
        ax.set_ylabel(r"$z$ (ft)", fontsize=11, fontweight="bold")
        ax.set_aspect("equal")

    fig.suptitle(
        "Two-Rockets BRT: Quasi-Linearized Cole-Hopf (NumPy)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved -> {save_path}")
    plt.close(fig)


# ============================================================================
#  Main entry point
# ============================================================================

if __name__ == "__main__":
    import os
    import time

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    cfg = SolverConfig(
        delta=0.08,
        num_samples=8_000,
        max_iters=20,
        tol=1e-5,
        t_start=0.0,
        t_end=1.0,
        seed=123,
    )

    solver = RocketsSolver(config=cfg)

    print("=" * 60)
    print("Two-Rockets Pursuit-Evasion BRT (Pure Python/NumPy)")
    print("=" * 60)

    t0 = time.time()
    plot_2d_slices(
        solver,
        theta_slices=[-pi / 4, 0.0, pi / 4],
        grid_res=25,
        domain=(-5.0, 5.0),
        save_path=os.path.join(results_dir, "rocket_2d_slices.png"),
    )
    elapsed = time.time() - t0
    print(f"Total time (NumPy): {elapsed:.1f}s")
