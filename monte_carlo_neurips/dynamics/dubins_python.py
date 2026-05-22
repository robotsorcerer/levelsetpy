#!/usr/bin/env python
"""3D Dubins Car Backward Reachability: Pure Python/NumPy Implementation.

Implements the quasi-linearization algorithm (Algorithm 1) from:

    "Approximately Correct and Scalable HJ-Reachability: A Sampling Scheme"
    ICML 2026

for a 3D Dubins vehicle model.

Problem Setup
-------------
The Dubins car is a kinematic model of a vehicle constrained to move at
bounded speeds with bounded turn rate:

State: x = (x, y, theta)
  - (x, y) in R^2: planar position
  - theta in [-pi, pi]: heading angle

Dynamics:
    x_dot     = v cos(theta)
    y_dot     = v sin(theta)
    theta_dot = omega

Control bounds:
    v in [v_min, v_max]
    omega in [-omega_max, omega_max]

For the pursuit-evasion formulation (two identical Dubins cars in relative
coordinates, following Merz 1972), the state represents the relative
position and heading of the evader w.r.t. the pursuer:

    x1_dot = -v_e + v_p cos(x3) + w_e x2
    x2_dot = -v_p sin(x3) - w_e x1
    x3_dot = -w_p - w_e

The Hamiltonian for the simple (single-player, minimizing) Dubins car:
    H(x, p) = v cos(theta) p1 + v sin(theta) p2 - omega_max |p3|

The Hamiltonian for the pursuit-evasion Dubins (Merz formulation):
    H(x, p) = p1(v_e - v_p cos(x3)) - p2(v_p sin(x3))
              - w|p1 x2 - p2 x1 - p3| + w|p3|

Terminal cost: Distance to target set (cylinder along theta axis).
"""

import numpy as np
from math import pi
from typing import Tuple, List, Optional, Callable
from functools import partial

from src.sampling_engine import (
    SolverConfig,
    compute_frozen_coefficient,
    relative_residual,
    mc_value_at_point,
    mc_gradient_at_point,
    mc_value_batch,
    mc_gradient_batch,
    solve_quasi_linear,
    cylinder_cost,
    sphere_cost,
)


# ============================================================================
#  Dubins Car Hamiltonians
# ============================================================================

class DubinsHamiltonian:
    """Simple Dubins car Hamiltonian (single-player, minimizing).

    Dynamics: x_dot = v cos(theta), y_dot = v sin(theta), theta_dot = omega
    Control: omega in [-omega_max, omega_max]

    The HJI Hamiltonian for backward reachability (target reaching):
        H(x, p) = v cos(theta) p1 + v sin(theta) p2 - omega_max |p3|

    The optimal control is bang-bang: omega* = -omega_max sign(p3).

    Parameters
    ----------
    speed : float
        Constant forward speed v.
    omega_max : float
        Maximum angular rate.
    smoothing_eps : float
        Smoothing for |p3| approximation.
    """

    def __init__(
        self,
        speed: float = 1.0,
        omega_max: float = 1.0,
        smoothing_eps: float = 1e-4,
    ):
        self.speed = speed
        self.omega_max = omega_max
        self.eps = smoothing_eps

    def __call__(self, t: float, x: np.ndarray, p: np.ndarray) -> float:
        p1, p2, p3 = p[0], p[1], p[2]
        theta = x[2]
        abs_p3 = np.sqrt(p3 ** 2 + self.eps ** 2)
        return float(
            self.speed * np.cos(theta) * p1
            + self.speed * np.sin(theta) * p2
            - self.omega_max * abs_p3
        )

    @property
    def state_dim(self) -> int:
        return 3


class DubinsRelativeHamiltonian:
    """Dubins pursuit-evasion Hamiltonian in relative coordinates (Merz 1972).

    Two identical Dubins vehicles. State: relative position and heading.

    Dynamics:
        x1_dot = -v_e + v_p cos(x3) + w_e x2
        x2_dot = -v_p sin(x3) - w_e x1
        x3_dot = -w_p - w_e

    Hamiltonian (pursuer minimizes, evader maximizes):
        H(x, p) = p1(v_e - v_p cos(x3)) - p2(v_p sin(x3))
                  - w|p1 x2 - p2 x1 - p3| + w|p3|

    This matches the levelsetpy DubinsVehicleRel implementation.

    Parameters
    ----------
    v_p : float
        Pursuer speed (negative convention to match levelsetpy).
    v_e : float
        Evader speed.
    w : float
        Angular speed bound (equal for both vehicles).
    smoothing_eps : float
        Smoothing for |.| approximation.
    """

    def __init__(
        self,
        v_p: float = -1.0,
        v_e: float = 1.0,
        w: float = 1.0,
        smoothing_eps: float = 1e-4,
    ):
        self.v_p = v_p
        self.v_e = v_e
        self.w = w
        self.eps = smoothing_eps

    def __call__(self, t: float, x: np.ndarray, p: np.ndarray) -> float:
        p1, p2, p3 = p[0], p[1], p[2]
        x1, x2, x3 = x[0], x[1], x[2]

        smooth_abs = lambda z: np.sqrt(z ** 2 + self.eps ** 2)

        term1 = p1 * (self.v_e - self.v_p * np.cos(x3))
        term2 = -p2 * (self.v_p * np.sin(x3))
        term3 = -self.w * smooth_abs(p1 * x2 - p2 * x1 - p3)
        term4 = self.w * smooth_abs(p3)
        return float(term1 + term2 + term3 + term4)

    @property
    def state_dim(self) -> int:
        return 3


# ============================================================================
#  Terminal cost
# ============================================================================

def dubins_terminal_cost(
    x: np.ndarray,
    radius: float = 0.5,
    target_center: Optional[np.ndarray] = None,
) -> float:
    """Terminal cost for Dubins car: cylinder signed distance.

    g(x) = ||(x, y)|| - radius  (ignores theta dimension).
    """
    return cylinder_cost(x, axis_align=2, radius=radius,
                         center=target_center)


# ============================================================================
#  Dubins solver
# ============================================================================

class DubinsSolver:
    """Complete solver for the Dubins car backward reachability problem.

    Supports both the simple (single-player) and pursuit-evasion
    (two-player, Merz) formulations.

    Parameters
    ----------
    config : SolverConfig
        Solver parameters.
    mode : str
        'simple' for single-player, 'pursuit_evasion' for two-player.
    speed : float
        Vehicle speed (or pursuer speed for PE).
    omega_max : float
        Angular rate bound.
    target_radius : float
        Capture cylinder radius.
    """

    def __init__(
        self,
        config: Optional[SolverConfig] = None,
        mode: str = "pursuit_evasion",
        speed: float = 1.0,
        omega_max: float = 1.0,
        target_radius: float = 0.5,
    ):
        self.cfg = config or SolverConfig(
            delta=0.08,
            num_samples=10_000,
            max_iters=15,
            tol=1e-5,
            t_start=0.0,
            t_end=1.0,
            seed=42,
        )
        self.mode = mode
        self.target_radius = target_radius

        if mode == "simple":
            self.H = DubinsHamiltonian(
                speed=speed, omega_max=omega_max,
            )
        elif mode == "pursuit_evasion":
            self.H = DubinsRelativeHamiltonian(
                v_p=-speed, v_e=speed, w=omega_max,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.g_fn = partial(dubins_terminal_cost, radius=target_radius)
        self.rng = np.random.default_rng(self.cfg.seed)

    def _compute_initial_c(
        self, eval_points: np.ndarray, t: float,
    ) -> np.ndarray:
        """Compute c^{(0)} = 2 H(t;x,Dg) / (delta |Dg|^2)."""
        M = eval_points.shape[0]
        c_init = np.zeros(M)

        for i in range(M):
            xi = eval_points[i]
            x_pos, y_pos = xi[0], xi[1]
            r = np.sqrt(x_pos ** 2 + y_pos ** 2)
            r_safe = max(r, 1e-10)
            Dg = np.array([x_pos / r_safe, y_pos / r_safe, 0.0])

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
        grid_res: int = 40,
        domain: Tuple[float, float] = (-4.0, 4.0),
        t_eval: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[float]]:
        """Solve on a 2D (x, y) slice at fixed theta.

        Returns
        -------
        X, Y : np.ndarray, shape (grid_res, grid_res)
        V : np.ndarray, shape (grid_res, grid_res)
        history : list of float
        """
        xs = np.linspace(domain[0], domain[1], grid_res)
        X, Y = np.meshgrid(xs, xs, indexing="ij")

        theta_col = np.full((grid_res * grid_res, 1), theta_val)
        eval_points = np.concatenate(
            [X.ravel()[:, np.newaxis], Y.ravel()[:, np.newaxis], theta_col],
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
        return X, Y, V, history

    def solve_volume(
        self,
        grid_res: int = 25,
        spatial_domain: Tuple[float, float] = (-4.0, 4.0),
        theta_range: Tuple[float, float] = (-pi, pi),
        t_eval: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[float]]:
        """Solve on the full 3D (x, y, theta) grid."""
        xs = np.linspace(spatial_domain[0], spatial_domain[1], grid_res)
        ts = np.linspace(theta_range[0], theta_range[1], grid_res)
        XX, YY, TT = np.meshgrid(xs, xs, ts, indexing="ij")

        eval_points = np.stack(
            [XX.ravel(), YY.ravel(), TT.ravel()], axis=-1,
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
        return XX, YY, TT, V, history


# ============================================================================
#  Plotting
# ============================================================================

def plot_2d_slices(
    solver: DubinsSolver,
    theta_slices: Optional[List[float]] = None,
    grid_res: int = 40,
    domain: Tuple[float, float] = (-4.0, 4.0),
    save_path: Optional[str] = None,
) -> None:
    """Plot 2D (x, y) slices of the Dubins BRT."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if theta_slices is None:
        theta_slices = [-pi / 2, 0.0, pi / 2]

    n_slices = len(theta_slices)
    fig, axes = plt.subplots(
        1, n_slices, figsize=(4 * n_slices, 4),
        sharex=True, sharey=True,
    )
    if n_slices == 1:
        axes = [axes]

    for ax, theta_val in zip(axes, theta_slices):
        print(f"  Solving theta = {theta_val:.3f} ...")
        X, Y, V, history = solver.solve_slice(
            theta_val, grid_res=grid_res, domain=domain,
        )

        ax.contourf(X, Y, V, levels=20, cmap="RdBu_r")
        ax.contour(X, Y, V, levels=[0.0], colors="k", linewidths=2)
        ax.set_title(
            rf"$\theta = {theta_val:.2f}$  ({len(history)} iters)",
            fontsize=11, fontweight="bold",
        )
        ax.set_xlabel(r"$x_1$ (m)", fontsize=11, fontweight="bold")
        ax.set_ylabel(r"$x_2$ (m)", fontsize=11, fontweight="bold")
        ax.set_aspect("equal")

    fig.suptitle(
        f"Dubins Car BRT: Quasi-Linearized Cole-Hopf (NumPy)\nMode: {solver.mode}",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved -> {save_path}")
    plt.close(fig)


# ============================================================================
#  Main
# ============================================================================

if __name__ == "__main__":
    import os
    import time

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    cfg = SolverConfig(
        delta=0.08,
        num_samples=4_000,
        max_iters=12,
        tol=1e-5,
        t_start=0.0,
        t_end=1.0,
        seed=42,
    )

    # Pursuit-evasion formulation
    solver = DubinsSolver(config=cfg, mode="pursuit_evasion")

    print("=" * 60)
    print("Dubins Car BRT (Pure Python/NumPy)")
    print("=" * 60)

    t0 = time.time()
    plot_2d_slices(
        solver,
        theta_slices=[-pi / 2, 0.0, pi / 2],
        grid_res=25,
        domain=(-4.0, 4.0),
        save_path=os.path.join(results_dir, "dubins_2d_slices.png"),
    )
    elapsed = time.time() - t0
    print(f"Total time (NumPy): {elapsed:.1f}s")
