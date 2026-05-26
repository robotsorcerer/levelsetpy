"""Visualization and disk-output handler for murmuration BRT experiments.

Saves trajectories, value-function heatmaps, phase diagrams, reachability
contours, topology summaries, and phase-transition logs to a configurable
output directory (default /tmp/murmurations/).
"""

import os
from typing import List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # headless backend for scripts
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False


class OutputHandler:
    """Writes all murmuration visualisation artefacts to *output_dir*."""

    def __init__(self, output_dir: str = "/tmp/murmurations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._phase_log_path = os.path.join(output_dir, "phase_transitions.txt")

    # ------------------------------------------------------------------
    # 1. Trajectory plot
    # ------------------------------------------------------------------

    def save_trajectory(
        self,
        t: int,
        x_traj: np.ndarray,
        flock_ids: np.ndarray,
    ) -> str:
        """Save 2D trajectory scatter colored by flock ID.

        Parameters
        ----------
        t : int
            Time-step index (used as frame counter in filename).
        x_traj : np.ndarray, shape (N, >=2)
            Agent positions; columns 0 and 1 are (x1, x2).
        flock_ids : np.ndarray, shape (N,)
            Integer flock identifier per agent.

        Returns
        -------
        str
            Absolute path of the saved figure.
        """
        if not _MPL_AVAILABLE:
            return ""
        fig, ax = plt.subplots(figsize=(10, 8))
        unique_ids = np.unique(flock_ids)
        cmap = plt.get_cmap("tab10", max(len(unique_ids), 1))
        for i, fid in enumerate(unique_ids):
            mask = flock_ids == fid
            ax.scatter(
                x_traj[mask, 0],
                x_traj[mask, 1],
                s=8,
                alpha=0.6,
                color=cmap(i),
                label=f"Flock {fid}",
            )
        ax.set_xlabel("x1 (m)")
        ax.set_ylabel("x2 (m)")
        ax.set_title(f"Agent trajectories  (t={t:04d})")
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        path = os.path.join(self.output_dir, f"trajectory_t{t:04d}.jpg")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # 2. Value-function heatmap
    # ------------------------------------------------------------------

    def save_heatmap(
        self,
        t: int,
        v_grid: np.ndarray,
        x_extent: Tuple[float, float, float, float] = (-5.0, 5.0, -5.0, 5.0),
    ) -> str:
        """Save value function v(x,t) as a heatmap.

        Parameters
        ----------
        t : int
            Time-step index.
        v_grid : np.ndarray, shape (H, W)
            2D slice of the value function on a regular grid.
        x_extent : (x1_min, x1_max, x2_min, x2_max)
            Physical extents for axis labels.

        Returns
        -------
        str
            Absolute path of the saved figure.
        """
        if not _MPL_AVAILABLE:
            return ""
        fig, ax = plt.subplots(figsize=(10, 8))
        x1_min, x1_max, x2_min, x2_max = x_extent
        im = ax.imshow(
            v_grid,
            extent=[x1_min, x1_max, x2_min, x2_max],
            origin="lower",
            cmap="viridis",
            aspect="auto",
        )
        plt.colorbar(im, ax=ax, label="v(x, t)")
        ax.contour(
            v_grid,
            levels=[0.0],
            colors="red",
            linewidths=1.5,
            extent=[x1_min, x1_max, x2_min, x2_max],
        )
        ax.set_xlabel("x1 (m)")
        ax.set_ylabel("x2 (m)")
        ax.set_title(f"Value function heatmap  (t={t:04d})")
        path = os.path.join(self.output_dir, f"heatmap_t{t:04d}.jpg")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # 3. Phase diagram (relative position / heading)
    # ------------------------------------------------------------------

    def save_phase_diagram(
        self,
        t: int,
        x_traj: np.ndarray,
        flock_ids: np.ndarray,
    ) -> str:
        """Phase space: relative position vs. heading difference.

        Plots (x1, x2) colored by heading angle x[3].

        Parameters
        ----------
        t : int
            Time-step index.
        x_traj : np.ndarray, shape (N, 4)
            Agent states (x1, x2, x3, theta).
        flock_ids : np.ndarray, shape (N,)
            Integer flock identifier per agent.

        Returns
        -------
        str
            Absolute path of the saved figure.
        """
        if not _MPL_AVAILABLE or x_traj.shape[1] < 4:
            return ""
        fig, ax = plt.subplots(figsize=(9, 8))
        sc = ax.scatter(
            x_traj[:, 0],
            x_traj[:, 1],
            c=x_traj[:, 3],
            cmap="hsv",
            s=6,
            alpha=0.5,
            vmin=-np.pi,
            vmax=np.pi,
        )
        plt.colorbar(sc, ax=ax, label="Heading θ (rad)")
        ax.set_xlabel("Relative x1 (m)")
        ax.set_ylabel("Relative x2 (m)")
        ax.set_title(f"Phase diagram: position vs. heading  (t={t:04d})")
        ax.grid(True, alpha=0.3)
        path = os.path.join(self.output_dir, f"phase_diagram_t{t:04d}.jpg")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # 4. Reachability contours (level sets of v)
    # ------------------------------------------------------------------

    def save_reachability(
        self,
        t: int,
        v_grid: np.ndarray,
        x_extent: Tuple[float, float, float, float] = (-5.0, 5.0, -5.0, 5.0),
        levels: Optional[List[float]] = None,
    ) -> str:
        """Save level-set contours of v(x, t) (reachability boundary).

        Parameters
        ----------
        t : int
            Time-step index.
        v_grid : np.ndarray, shape (H, W)
            2D slice of the value function.
        x_extent : tuple
            Physical extents (x1_min, x1_max, x2_min, x2_max).
        levels : list of float, optional
            Level-set values to draw; defaults to [-0.5, 0.0, 0.5].

        Returns
        -------
        str
            Absolute path of the saved figure.
        """
        if not _MPL_AVAILABLE:
            return ""
        if levels is None:
            levels = [-0.5, 0.0, 0.5]
        x1_min, x1_max, x2_min, x2_max = x_extent
        fig, ax = plt.subplots(figsize=(9, 8))
        x1_grid = np.linspace(x1_min, x1_max, v_grid.shape[1])
        x2_grid = np.linspace(x2_min, x2_max, v_grid.shape[0])
        cs = ax.contour(x1_grid, x2_grid, v_grid, levels=levels, cmap="plasma")
        ax.clabel(cs, inline=True, fontsize=9)
        ax.set_xlabel("x1 (m)")
        ax.set_ylabel("x2 (m)")
        ax.set_title(f"Reachability level sets  (t={t:04d})")
        ax.grid(True, alpha=0.3)
        path = os.path.join(self.output_dir, f"reachability_t{t:04d}.jpg")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # 5. Topology summary (single combined plot)
    # ------------------------------------------------------------------

    def save_topology_summary(
        self,
        t_range: np.ndarray,
        chi_list: List[float],
        beta1_list: List[float],
        n_comp_list: List[int],
    ) -> str:
        """Save χ(t), β₁(t), n_comp(t) vs. time as a single three-panel plot.

        Parameters
        ----------
        t_range : np.ndarray, shape (T,)
            Time-step indices.
        chi_list : list of float, length T
            Euler characteristic χ per time step.
        beta1_list : list of float, length T
            First Betti number β₁ per time step.
        n_comp_list : list of int, length T
            Number of BRT connected components per time step.

        Returns
        -------
        str
            Absolute path of the saved figure.
        """
        if not _MPL_AVAILABLE:
            return ""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(t_range, chi_list, "b-o", markersize=4)
        axes[0].set_ylabel("Euler characteristic χ(t)")
        axes[0].set_xlabel("Time step t")
        axes[0].set_title("Euler characteristic")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(t_range, beta1_list, "r-s", markersize=4)
        axes[1].set_ylabel("Betti number β₁(t)")
        axes[1].set_xlabel("Time step t")
        axes[1].set_title("First Betti number (holes)")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(t_range, n_comp_list, "g-^", markersize=4)
        axes[2].set_ylabel("n_components(t)")
        axes[2].set_xlabel("Time step t")
        axes[2].set_title("Connected components")
        axes[2].grid(True, alpha=0.3)

        fig.suptitle("BRT Topology Evolution", fontsize=13, fontweight="bold")
        fig.tight_layout()
        path = os.path.join(self.output_dir, "topology_summary.jpg")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # 6. Phase-transition text log
    # ------------------------------------------------------------------

    def log_phase_transitions(
        self,
        t: int,
        event_type: str,
        chi: float,
        beta1: float,
        n_comp: int,
    ) -> None:
        """Append one event line to phase_transitions.txt.

        Parameters
        ----------
        t : int
            Time-step index.
        event_type : str
            Name of the topological event (e.g. 'vacuole_nucleation').
        chi : float
            Euler characteristic at this step.
        beta1 : float
            First Betti number at this step.
        n_comp : int
            Number of connected components at this step.
        """
        with open(self._phase_log_path, "a") as fh:
            fh.write(
                f"{t:4d}  {event_type:25s}  "
                f"chi={chi:7.3f}  beta1={beta1:7.3f}  n_comp={n_comp:3d}\n"
            )
