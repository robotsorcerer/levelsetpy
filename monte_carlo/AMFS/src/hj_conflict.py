"""HJ-Gauss dynamics-aware conflict predicate for MAPF (AMFS innovations #1, #2).

Innovation #1 (windowed reachability == RHCR window): we solve the pairwise
collision Backward Reachable Tube (BRT) over a FINITE horizon T = w (the RHCR
rolling window). A relative state whose value is <= 0 at horizon w is one from
which collision is unavoidable within the window under worst-case behavior.

Innovation #2 (dynamics-aware conflict predicate): two agents are "in conflict"
iff their *relative Dubins state* lies inside this BRT. This replaces the
dynamics-blind grid/disc conflict test used by classical CBS/RHCR.

Two-phase design so the online MAPF loop and the multiseed harness stay
numpy-only (no JAX/scipy in the loop):
  * precompute_brt(...)  -- ONE-TIME, uses the JAX HJReachabilitySampler
    (Algorithm 1 in hjgauss.pdf) to solve the Dubins pursuit-evasion BRT on a
    relative-state grid; caches {axes, V} to an .npz.
  * BRTPredicate         -- loads the .npz and answers membership with a
    hand-rolled, vectorized trilinear interpolation (periodic in heading),
    numpy only.

The relative coordinates match src/hamiltonians/dubins_relative.py (Merz 1972):
  x1 =  cos(th_p) dx + sin(th_p) dy      (evader relative to pursuer, along-heading)
  x2 = -sin(th_p) dx + cos(th_p) dy      (cross-heading)
  x3 =  th_e - th_p                      (relative heading), wrapped to [-pi, pi)
where (dx,dy) = (x_e - x_p, y_e - y_p).
"""

from __future__ import annotations

import os
import numpy as np

_PI = np.pi
_TWO_PI = 2.0 * np.pi


# ────────────────────────────────────────────────────────────────────────────
#  Relative-coordinate transform (numpy, vectorized)
# ────────────────────────────────────────────────────────────────────────────
def wrap_angle(a):
    """Wrap angle(s) to [-pi, pi)."""
    return (np.asarray(a) + _PI) % _TWO_PI - _PI


def relative_state(pose_p, pose_e):
    """Relative Dubins state of evader w.r.t. pursuer.

    Parameters
    ----------
    pose_p, pose_e : array-like (..., 3) as (x, y, theta) in world frame.

    Returns
    -------
    (..., 3) array (x1, x2, x3) in the pursuer body frame.
    """
    pose_p = np.asarray(pose_p, dtype=float)
    pose_e = np.asarray(pose_e, dtype=float)
    dx = pose_e[..., 0] - pose_p[..., 0]
    dy = pose_e[..., 1] - pose_p[..., 1]
    thp = pose_p[..., 2]
    c, s = np.cos(thp), np.sin(thp)
    x1 = c * dx + s * dy
    x2 = -s * dx + c * dy
    x3 = wrap_angle(pose_e[..., 2] - thp)
    return np.stack([x1, x2, x3], axis=-1)


# ────────────────────────────────────────────────────────────────────────────
#  BRT precompute (uses the JAX sampler — Algorithm 1). ONE-TIME.
# ────────────────────────────────────────────────────────────────────────────
def precompute_brt(
    out_path: str,
    *,
    window_horizon: float = 0.6,
    v_p: float = -1.0,
    v_e: float = 1.0,
    w: float = 1.0,
    capture_radius: float = 0.8,
    delta: float = 0.1,
    num_samples: int = 3000,
    max_quasi_iters: int = 10,
    quasi_tol: float = 1e-4,
    x_lim: float = 4.0,
    n_xy: int = 31,
    n_theta: int = 21,
    seed: int = 0,
    monte_carlo_root: str | None = None,
    verbose: bool = True,
):
    """Solve the pairwise Dubins-PE BRT at horizon w and cache to `out_path`.

    Requires JAX (run under AMFS/.venv). The value is evaluated at t =
    window_horizon (innovation #1: reachability window = RHCR window w).
    """
    import sys
    import time

    if monte_carlo_root is None:
        # .../monte_carlo/AMFS/src/hj_conflict.py -> .../monte_carlo
        monte_carlo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
    if monte_carlo_root not in sys.path:
        sys.path.insert(0, monte_carlo_root)

    import jax
    jax.config.update("jax_platform_name", "cpu")
    import jax.numpy as jnp
    from functools import partial
    from src.config import SolverConfig
    from src.hamiltonians import DubinsRelativeHamiltonian
    from src.initial_conditions import cylinder_cost
    from src.hj_sampler import HJReachabilitySampler

    cfg = SolverConfig(
        delta=delta, num_samples=num_samples, max_quasi_iters=max_quasi_iters,
        quasi_tol=quasi_tol, t_start=0.0, t_end=1.0, seed=seed,
        chunk_size=max(2000, n_xy * n_xy),
    )
    H = DubinsRelativeHamiltonian(v_p=v_p, v_e=v_e, w=w)
    g = partial(cylinder_cost, axis_align=2, radius=capture_radius)

    x1_axis = np.linspace(-x_lim, x_lim, n_xy)
    x2_axis = np.linspace(-x_lim, x_lim, n_xy)
    # periodic theta grid: endpoint excluded so -pi and +pi are not duplicated
    th_axis = np.linspace(-_PI, _PI, n_theta, endpoint=False)

    X1, X2, TH = np.meshgrid(x1_axis, x2_axis, th_axis, indexing="ij")
    pts = jnp.asarray(
        np.stack([X1.ravel(), X2.ravel(), TH.ravel()], axis=-1), dtype=jnp.float32
    )

    if verbose:
        print(f"[precompute_brt] solving {pts.shape[0]} states, horizon w={window_horizon} ...")
    t0 = time.time()
    sampler = HJReachabilitySampler(H, g, cfg)
    v, hist = sampler.solve_quasi_linear(pts, float(window_horizon))
    V = np.asarray(v, dtype=np.float32).reshape(n_xy, n_xy, n_theta)
    dt = time.time() - t0
    if verbose:
        res = [h for h in hist if h == h]  # drop nans
        print(f"[precompute_brt] done in {dt:.1f}s, iters={len(hist)}, "
              f"final_residual={res[-1] if res else float('nan'):.2e}, "
              f"frac_inside={(V <= 0).mean():.3f}")

    meta = dict(
        window_horizon=window_horizon, v_p=v_p, v_e=v_e, w=w,
        capture_radius=capture_radius, delta=delta, num_samples=num_samples,
        max_quasi_iters=max_quasi_iters, quasi_tol=quasi_tol,
        x_lim=x_lim, n_xy=n_xy, n_theta=n_theta, seed=seed,
        solve_seconds=dt,
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez_compressed(
        out_path, V=V, x1_axis=x1_axis, x2_axis=x2_axis, th_axis=th_axis,
        meta=np.array([repr(meta)], dtype=object),
    )
    if verbose:
        print(f"[precompute_brt] cached -> {out_path}")
    return out_path


# ────────────────────────────────────────────────────────────────────────────
#  Runtime predicate (numpy only)
# ────────────────────────────────────────────────────────────────────────────
class BRTPredicate:
    """Numpy-only windowed BRT membership predicate over cached V(x1,x2,theta).

    conflict(pose_p, pose_e) := value(relative_state) <= margin.
    A positive `margin` inflates the certified set (extra conservatism / safety
    band, cf. the conservative-certificate corollary in the paper).
    """

    def __init__(self, npz_path: str, margin: float = 0.0):
        d = np.load(npz_path, allow_pickle=True)
        self.V = np.asarray(d["V"], dtype=float)
        self.x1_axis = np.asarray(d["x1_axis"], dtype=float)
        self.x2_axis = np.asarray(d["x2_axis"], dtype=float)
        self.th_axis = np.asarray(d["th_axis"], dtype=float)
        try:
            self.meta = eval(str(d["meta"][0]), {"__builtins__": {}})
        except Exception:
            self.meta = {}
        self.margin = float(margin)
        self.capture_radius = float(self.meta.get("capture_radius", 0.8))
        self.x_lim = float(self.x1_axis[-1])
        self._dth = self.th_axis[1] - self.th_axis[0]

    # -- vectorized periodic trilinear interpolation --------------------------
    def value(self, rel_states):
        """Interpolated BRT value at relative states (..., 3).

        Outside the (x1,x2) box the value is clamped to the nearest edge value
        (states far outside the box are trivially collision-free, value>0).
        Heading is periodic.
        """
        r = np.asarray(rel_states, dtype=float)
        flat = r.reshape(-1, 3)
        x1, x2, th = flat[:, 0], flat[:, 1], flat[:, 2]

        # clamp spatial coords into the grid box
        x1 = np.clip(x1, self.x1_axis[0], self.x1_axis[-1])
        x2 = np.clip(x2, self.x2_axis[0], self.x2_axis[-1])
        th = wrap_angle(th)

        def idx_frac(coord, axis):
            n = axis.size
            pos = (coord - axis[0]) / (axis[-1] - axis[0]) * (n - 1)
            i0 = np.clip(np.floor(pos).astype(int), 0, n - 2)
            f = pos - i0
            return i0, f

        i0, fi = idx_frac(x1, self.x1_axis)
        j0, fj = idx_frac(x2, self.x2_axis)
        # periodic theta
        nT = self.th_axis.size
        posk = (th - self.th_axis[0]) / self._dth
        k0 = np.floor(posk).astype(int) % nT
        fk = posk - np.floor(posk)
        k1 = (k0 + 1) % nT
        i1 = np.clip(i0 + 1, 0, self.x1_axis.size - 1)
        j1 = np.clip(j0 + 1, 0, self.x2_axis.size - 1)

        V = self.V
        def g(a, b, c):
            return V[a, b, c]

        c00 = g(i0, j0, k0) * (1 - fi) + g(i1, j0, k0) * fi
        c10 = g(i0, j1, k0) * (1 - fi) + g(i1, j1, k0) * fi
        c01 = g(i0, j0, k1) * (1 - fi) + g(i1, j0, k1) * fi
        c11 = g(i0, j1, k1) * (1 - fi) + g(i1, j1, k1) * fi
        c0 = c00 * (1 - fj) + c10 * fj
        c1 = c01 * (1 - fj) + c11 * fj
        out = c0 * (1 - fk) + c1 * fk
        return out.reshape(r.shape[:-1])

    def in_conflict(self, pose_p, pose_e):
        """Boolean conflict: relative state inside the (margin-inflated) BRT."""
        rel = relative_state(pose_p, pose_e)
        return self.value(rel) <= self.margin
