"""Executor-coordinate reachability tables for compatible steering control.

Unlike the historical caches, the first vehicle is the defending ego vehicle.
The table is a numerical candidate barrier, not a verified safety certificate.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from hj_conflict import relative_state, wrap_angle

MODEL = "ego_body_defender_max_other_min_v1"


def relative_dynamics(q, ego_turn, other_turn, ego_speed=1.0, other_speed=1.0):
    """Derivative of relative_state(ego, other), without disturbance."""
    q = np.asarray(q, dtype=float)
    x, y, phi = np.moveaxis(q, -1, 0)
    return np.stack((other_speed * np.cos(phi) - ego_speed + ego_turn * y,
                     other_speed * np.sin(phi) - ego_turn * x,
                     np.broadcast_to(other_turn - ego_turn, x.shape)), axis=-1)


class ReachabilityTable:
    """Periodic trilinear value and its within-cell analytic gradient.

    Spatial extrapolation is forbidden. Grid faces have several limiting
    gradients: gradient_branches returns all incident-cell gradients there.
    Tables have JSON metadata and never require pickle deserialization.
    """

    def __init__(self, path, margin=0.0):
        with np.load(path, allow_pickle=False) as data:
            if "metadata_json" not in data:
                raise ValueError("Legacy BRT cache: regenerate with precompute_compatible_brt.py")
            self.meta = json.loads(str(data["metadata_json"].item()))
            self.V = np.asarray(data["V"], dtype=float)
            self.axes = tuple(np.asarray(data[k], dtype=float)
                              for k in ("x1_axis", "x2_axis", "th_axis"))
        if self.meta.get("model") != MODEL:
            raise ValueError("Incompatible BRT coordinates; regenerate with precompute_compatible_brt.py")
        if self.V.shape != tuple(len(a) for a in self.axes) or not np.isfinite(self.V).all():
            raise ValueError("BRT must be a finite array matching its axes")
        for axis in self.axes:
            if len(axis) < 3 or not np.isfinite(axis).all() or not np.all(np.diff(axis) > 0):
                raise ValueError("BRT axes must be finite and strictly increasing")
            if not np.allclose(np.diff(axis), np.diff(axis)[0]):
                raise ValueError("BRT axes must be uniformly spaced")
        if not np.isclose(self.axes[2][0], -np.pi) or not np.isclose(
                len(self.axes[2]) * np.diff(self.axes[2])[0], 2 * np.pi):
            raise ValueError("Heading grid must cover [-pi, pi) without a repeated endpoint")
        self.spacing = np.array([a[1] - a[0] for a in self.axes])
        self.margin = float(margin)
        self.capture_radius = float(self.meta["capture_radius"])
        for key in ("capture_radius", "speed", "turn_rate", "horizon"):
            if not np.isfinite(self.meta[key]) or self.meta[key] <= 0:
                raise ValueError(f"Invalid BRT metadata: {key}")
        if not np.isfinite(self.margin) or self.margin < 0:
            raise ValueError("margin must be finite and nonnegative")
        geometric = np.hypot(self.axes[0][:, None], self.axes[1][None, :]) - self.capture_radius
        if np.any(self.V > geometric[..., None]):
            raise ValueError("BRT values must not exceed the nodal contact-distance function")
        # The distance function is 1-Lipschitz. This bounds its trilinear
        # interpolation overestimate, independently of PDE solver accuracy.
        self.geometric_padding = 0.5 * float(np.hypot(*self.spacing[:2]))

    def contains(self, q):
        q = np.asarray(q, dtype=float)
        return (np.isfinite(q).all(axis=-1)
                & (q[..., 0] >= self.axes[0][0]) & (q[..., 0] <= self.axes[0][-1])
                & (q[..., 1] >= self.axes[1][0]) & (q[..., 1] <= self.axes[1][-1]))

    def _coordinates(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape[-1:] != (3,) or not np.all(self.contains(q)):
            raise ValueError("Relative state is nonfinite or outside the BRT spatial domain")
        q = q.copy()
        q[..., 2] = wrap_angle(q[..., 2])
        pos = (q - np.array([a[0] for a in self.axes])) / self.spacing
        cells = np.floor(pos).astype(int)
        for dim in (0, 1):
            cells[..., dim] = np.clip(cells[..., dim], 0, len(self.axes[dim]) - 2)
        return pos, cells

    def _interpolate(self, pos, cells, *, with_gradient=True):
        frac = pos - cells
        value = np.zeros(pos.shape[:-1])
        grad = np.zeros_like(pos) if with_gradient else None
        for i in (0, 1):
            for j in (0, 1):
                for k in (0, 1):
                    corner = np.array([i, j, k])
                    indices = cells + corner
                    v = self.V[indices[..., 0], indices[..., 1],
                               indices[..., 2] % len(self.axes[2])]
                    weights = np.where(corner, frac, 1 - frac)
                    value += v * np.prod(weights, axis=-1)
                    if with_gradient:
                        for dim in range(3):
                            other_dims = [d for d in range(3) if d != dim]
                            grad[..., dim] += (v * (2 * corner[dim] - 1)
                                              * np.prod(weights[..., other_dims], axis=-1)
                                              / self.spacing[dim])
        return value, grad

    def value_and_gradient(self, q):
        return self._interpolate(*self._coordinates(q))

    def gradient_branches(self, q):
        """All limiting gradients at one state; one gradient off grid faces."""
        q = np.asarray(q, dtype=float)
        if q.shape != (3,):
            raise ValueError("gradient_branches expects one relative state")
        pos, cells = self._coordinates(q)
        candidates = [cells]
        for dim in range(3):
            if np.isclose(pos[dim], round(pos[dim]), atol=1e-9, rtol=0):
                # Enumerate both sides even when roundoff makes floor(pos)
                # select the left cell at a nominally integral coordinate.
                node = int(round(pos[dim]))
                extra = []
                for index in (node - 1, node):
                    if dim == 2 or 0 <= index < len(self.axes[dim]) - 1:
                        for cell in candidates:
                            new = cell.copy()
                            new[dim] = index
                            extra.append(new)
                candidates = extra
        _, grad = self._interpolate(np.broadcast_to(pos, (len(candidates), 3)),
                                    np.array(candidates))
        return np.unique(grad, axis=0)

    def value(self, q):
        return self._interpolate(*self._coordinates(q), with_gradient=False)[0]

    def in_conflict(self, ego, other):
        q = relative_state(ego, other)
        if not np.all(self.contains(q)):
            # Legacy flee comparison: a far pair beyond the cache is ignored.
            # The compatible controller uses an explicit distance-bound check.
            return False
        return self.value(q) <= self.geometric_padding + self.margin


def solve_brt(path, *, horizon=0.6, capture_radius=0.5, speed=1.0,
              turn_rate=1.0, relative_disturbance=0.0, x_lim=4.0,
              n_xy=61, n_theta=48, cfl=0.8):
    """First-order local Lax--Friedrichs BRT solve, numpy/CPU only.

    V_tau = min(0, max_ego min_other grad(V).f), V(0)=distance-r.
    Optional relative disturbance is a Euclidean velocity bound in the body
    frame. Runtime world-axis box bounds are handled separately by the shield.
    This solver provides no certified uniform error or gradient-error bound.
    """
    if (not all(np.isfinite(x) and x > 0 for x in
                (horizon, capture_radius, speed, turn_rate, x_lim))
            or not np.isfinite(relative_disturbance) or relative_disturbance < 0
            or not 0 < cfl <= 1 or n_xy < 5 or n_theta < 8):
        raise ValueError("Invalid solver parameters")
    if x_lim <= capture_radius + (2 * speed + relative_disturbance) * horizon:
        raise ValueError("Spatial box must contain the finite-horizon collision-reachable ball")
    axes = (np.linspace(-x_lim, x_lim, n_xy), np.linspace(-x_lim, x_lim, n_xy),
            np.linspace(-np.pi, np.pi, n_theta, endpoint=False))
    x, y, phi = np.meshgrid(*axes, indexing="ij")
    value = np.hypot(x, y) - capture_radius
    initial = value.copy()
    ds = [a[1] - a[0] for a in axes]
    drift_x, drift_y = speed * (np.cos(phi) - 1), speed * np.sin(phi)
    alphas = (np.abs(drift_x) + turn_rate * np.abs(y) + relative_disturbance,
              np.abs(drift_y) + turn_rate * np.abs(x) + relative_disturbance,
              np.full_like(phi, 2 * turn_rate))
    dt_max = cfl / float(np.max(sum(a / d for a, d in zip(alphas, ds))))
    steps = int(np.ceil(horizon / dt_max))
    dt = horizon / steps
    start = time.perf_counter()
    for _ in range(steps):
        central, visc = [], np.zeros_like(value)
        for dim in range(3):
            ahead, behind = np.roll(value, -1, axis=dim), np.roll(value, 1, axis=dim)
            if dim < 2:
                lo, hi = [slice(None)] * 3, [slice(None)] * 3
                lo[dim], hi[dim] = 0, -1
                lo1, hi1 = lo.copy(), hi.copy()
                lo1[dim], hi1[dim] = 1, -2
                behind[tuple(lo)] = 2 * value[tuple(lo)] - value[tuple(lo1)]
                ahead[tuple(hi)] = 2 * value[tuple(hi)] - value[tuple(hi1)]
            plus = (ahead - value) / ds[dim]
            minus = (value - behind) / ds[dim]
            central.append(0.5 * (plus + minus))
            visc += 0.5 * alphas[dim] * (plus - minus)
        px, py, pt = central
        hamiltonian = (px * drift_x + py * drift_y
                       + turn_rate * np.abs(px * y - py * x - pt)
                       - turn_rate * np.abs(pt)
                       - relative_disturbance * np.hypot(px, py))
        value += dt * np.minimum(0.0, hamiltonian + visc)
    elapsed = time.perf_counter() - start
    if not np.isfinite(value).all() or np.any(value > initial + 1e-12):
        raise RuntimeError("BRT integration failed its finite/tube checks")
    metadata = dict(model=MODEL, solver="numpy-first-order-LLF", horizon=horizon,
                    capture_radius=capture_radius, speed=speed, turn_rate=turn_rate,
                    relative_disturbance=relative_disturbance, x_lim=x_lim,
                    n_xy=n_xy, n_theta=n_theta, cfl=cfl, steps=steps,
                    solve_seconds=elapsed, certified_error_bound=None)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, V=value, x1_axis=axes[0], x2_axis=axes[1],
                        th_axis=axes[2], metadata_json=json.dumps(metadata))
    return metadata
