"""BRT phase transition topology tracking.

Detects topological changes in the zero-level-set of the BRT as the system evolves.
Key events: vacuole nucleation, cordon formation, flock fragmentation.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional

import jax.numpy as jnp
import numpy as np
from scipy import ndimage


@dataclass
class TopologyState:
    """Topological invariants of a BRT zero-level-set."""

    n_components: int
    betti_1: int
    euler_char: float
    time_idx: int = 0

    @property
    def signature(self) -> Tuple[int, int, float]:
        """Return (n_components, betti_1, euler_char) as a hashable tuple."""
        return (self.n_components, self.betti_1, self.euler_char)


def brt_topology_signature(
    v_2d_slice: jnp.ndarray, threshold: float = 0.0
) -> TopologyState:
    """Compute topological invariants of BRT zero-level-set in 2D slice.

    Parameters
    ----------
    v_2d_slice : jnp.ndarray, shape (h, w)
        2D slice of the BRT value function v(x₁, x₂).
    threshold : float
        Level-set threshold (default 0). {x: v <= threshold} is the BRT.

    Returns
    -------
    TopologyState
        Topological signature at this time.
    """
    v_np = np.asarray(v_2d_slice)
    brt_set = v_np <= threshold  # Binary: True inside BRT

    # Connected components of BRT
    labeled_array, n_components = ndimage.label(brt_set)

    # Betti number β₁ = number of independent loops (holes)
    # For 2D, Euler characteristic χ = V - E + F
    # Approximation: count holes via erosion/dilation
    betti_1 = 0
    for comp_id in range(1, n_components + 1):
        comp_mask = labeled_array == comp_id
        # Erosion to shrink the component; if it becomes empty, it was a thin
        # structure. If it remains, the filled-in complement might have holes.
        complement = ~comp_mask
        filled_complement, n_holes = ndimage.label(complement)
        # If complement has >1 component, the original component has holes
        if n_holes > 1:
            betti_1 += n_holes - 1  # -1 for the outer infinite component

    # Euler characteristic: χ = V - E + F (for a graph/cell complex on grid)
    # Simplified: χ ≈ n_components - betti_1 (for planar graphs)
    euler_char = float(n_components - betti_1)

    return TopologyState(
        n_components=n_components, betti_1=betti_1, euler_char=euler_char
    )


def detect_phase_transitions(
    topology_history: List[TopologyState],
    v_slices: Optional[List[jnp.ndarray]] = None,
    flash_radius_jump: float = 2.0,
) -> List[Tuple[int, str]]:
    """Detect phase transition events from topology history.

    Parameters
    ----------
    topology_history : list of TopologyState
        Time-evolving topology signatures.
    v_slices : list of jnp.ndarray, optional
        2D value function slices (one per time step). When provided, flash
        expansion is also detected via a jump in BRT radius (primary criterion),
        in addition to the topological proxy (secondary criterion).
    flash_radius_jump : float
        Minimum increase in BRT radius (in grid units) to qualify as flash
        expansion when ``v_slices`` is provided.

    Returns
    -------
    events : list of (time_idx, event_name)
        List of detected events in chronological order.
    """
    events = []

    if len(topology_history) < 2:
        return events

    # Precompute radii if value slices available
    radii: List[float] = []
    if v_slices is not None:
        radii = [brt_radius_at_time(v) for v in v_slices]

    for i in range(1, len(topology_history)):
        prev = topology_history[i - 1]
        curr = topology_history[i]

        # Vacuole nucleation: Euler characteristic decreases (new hole opens)
        if curr.euler_char < prev.euler_char - 0.5:
            events.append((i, "vacuole_nucleation"))

        # Cordon formation: Betti number becomes 1 (annular BRT)
        if curr.betti_1 >= 1 and prev.betti_1 == 0:
            events.append((i, "cordon_formation"))

        # Flock fragmentation: number of components increases
        if curr.n_components > prev.n_components:
            events.append((i, "flock_fragmentation"))

        # Flash expansion: primary — large radial jump in BRT boundary;
        # secondary (topological proxy) — fragmented BRT re-unifies into a
        # single simply-connected component.
        flash_detected = False
        if radii and radii[i] - radii[i - 1] >= flash_radius_jump:
            flash_detected = True
        elif prev.n_components > 1 and curr.n_components == 1 and curr.betti_1 == 0:
            flash_detected = True
        if flash_detected:
            events.append((i, "flash_expansion"))

    return events


def animate_brt_evolution(
    v_slices_time: List[jnp.ndarray],
    topology_history: List[TopologyState],
    events: List[Tuple[int, str]],
    save_path: str = "brt_evolution.gif",
    dpi: int = 100,
    fps: int = 2,
):
    """Create animated visualization of BRT evolution with phase transition events.

    Parameters
    ----------
    v_slices_time : list of jnp.ndarray
        Value function 2D slices, one per time step. shape (n_times, h, w).
    topology_history : list of TopologyState
        Topology signature at each time step.
    events : list of (time_idx, event_name)
        Detected phase transition events.
    save_path : str
        Path to save the .gif animation.
    dpi : int
        DPI for the figure.
    fps : int
        Frames per second for GIF.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from matplotlib.colors import Normalize
    except ImportError:
        print("matplotlib not available; skipping animation")
        return

    n_times = len(v_slices_time)
    event_dict = {t: name for t, name in events}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=dpi)

    # Prepare frames
    def init():
        ax1.cla()
        ax2.cla()
        return []

    def animate_frame(frame_idx):
        ax1.cla()
        ax2.cla()

        # Left: contourf of v with zero-level-set highlighted
        v_slice = np.asarray(v_slices_time[frame_idx])
        levels = np.linspace(v_slice.min(), v_slice.max(), 20)
        cf = ax1.contourf(v_slice, levels=levels, cmap="RdBu_r")
        ax1.contour(v_slice, levels=[0], colors="black", linewidths=2)
        ax1.set_title(f"BRT zero-level-set (t={frame_idx})")
        ax1.set_xlabel("x₁ (m)")
        ax1.set_ylabel("x₂ (m)")
        plt.colorbar(cf, ax=ax1, label="v(x)")

        # Right: topology metrics over time
        times_so_far = np.arange(frame_idx + 1)
        n_comps = [topology_history[t].n_components for t in range(frame_idx + 1)]
        betti_1s = [topology_history[t].betti_1 for t in range(frame_idx + 1)]

        ax2.plot(times_so_far, n_comps, "b-o", label="n_components", markersize=4)
        ax2.plot(times_so_far, betti_1s, "r-s", label="β₁ (holes)", markersize=4)

        # Mark events
        for event_time, event_name in events:
            if event_time <= frame_idx:
                ax2.axvline(event_time, color="green", linestyle="--", alpha=0.5)
                ax2.text(
                    event_time,
                    ax2.get_ylim()[1] * 0.95,
                    event_name.replace("_", "\n"),
                    fontsize=8,
                    ha="center",
                    color="green",
                )

        ax2.set_xlabel("Time index")
        ax2.set_ylabel("Topology metric")
        ax2.set_title("BRT topology evolution")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        if frame_idx in event_dict:
            fig.suptitle(f"Event at t={frame_idx}: {event_dict[frame_idx]}", fontsize=12, color="red")

        return []

    anim = animation.FuncAnimation(
        fig, animate_frame, init_func=init, frames=n_times, interval=1000 // fps, blit=False
    )

    try:
        anim.save(save_path, writer="pillow", fps=fps)
        print(f"Saved animation to {save_path}")
    except Exception as e:
        print(f"Failed to save animation: {e}")

    plt.close(fig)


def brt_radius_at_time(v_2d_slice: jnp.ndarray, threshold: float = 0.0) -> float:
    """Estimate BRT radius (maximum distance from origin to zero-level-set).

    Parameters
    ----------
    v_2d_slice : jnp.ndarray, shape (h, w)
        2D slice of value function.
    threshold : float
        Level-set threshold.

    Returns
    -------
    float
        Estimated BRT radius (in grid units).
    """
    brt_set = np.asarray(v_2d_slice) <= threshold
    if not brt_set.any():
        return 0.0

    # Get coordinates of BRT points
    coords = np.argwhere(brt_set)  # (n_points, 2) in (row, col) = (x2_idx, x1_idx)
    center = np.array([brt_set.shape[0] / 2, brt_set.shape[1] / 2])
    distances = np.linalg.norm(coords - center, axis=1)
    return float(np.max(distances))
