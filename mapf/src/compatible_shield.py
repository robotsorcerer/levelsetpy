"""Intersect simultaneous pairwise steering constraints; numpy only.

For h=V(q)-padding-margin, require dot(h)+gain*h >= rate_margin.
Other robots' turn rates and both robots' translational disturbances are
minimized analytically. Each gradient branch gives an affine inequality in
the ego turn rate. Feasibility here is instantaneous numerical feasibility,
not a certificate for a sampled rollout. See COMPATIBLE_CONTROL.md.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from compatible_brt import ReachabilityTable
from hj_conflict import relative_state


@dataclass
class IntervalResult:
    control: float
    lower: float
    upper: float
    feasible: bool
    min_residual: float


def project_interval(nominal, slopes, offsets, turn_bound, *, tolerance=1e-10):
    """Project onto all b*u+c >= 0; if empty, maximize the minimum residual.

    The infeasible fallback is bounded and deterministic, but is not safe.
    A concave piecewise-affine lower envelope reaches its maximum at an
    endpoint, an intersection of two lines, or a flat interval (nominal wins
    ties). No row, including a zero-slope row, is silently discarded.
    """
    b, c = np.asarray(slopes, dtype=float), np.asarray(offsets, dtype=float)
    if (b.ndim != 1 or c.shape != b.shape or not np.isfinite(b).all()
            or not np.isfinite(c).all() or not np.isfinite(nominal)
            or not np.isfinite(turn_bound) or turn_bound <= 0):
        raise ValueError("Invalid affine constraints or turn bound")
    lo, hi = -float(turn_bound), float(turn_bound)
    positive, negative, zero = b > 0, b < 0, b == 0
    if positive.any():
        lo = max(lo, float(np.max(-c[positive] / b[positive])))
    if negative.any():
        hi = min(hi, float(np.min(-c[negative] / b[negative])))
    feasible = lo <= hi and not np.any(c[zero] < 0)
    if feasible:
        command = float(np.clip(nominal, lo, hi))
    else:
        candidates = [-turn_bound, turn_bound, float(np.clip(nominal, -turn_bound, turn_bound))]
        if b.size > 1:
            i, j = np.triu_indices(b.size, 1)
            different = b[i] != b[j]
            crossing = (c[j[different]] - c[i[different]]) / (b[i[different]] - b[j[different]])
            candidates.extend(crossing[(crossing >= -turn_bound) & (crossing <= turn_bound)])
        candidates = np.asarray(candidates)
        score = np.min(b[:, None] * candidates + c[:, None], axis=0)
        best = np.max(score)
        ties = np.flatnonzero(np.isclose(score, best, rtol=0, atol=tolerance))
        chosen = ties[np.argmin(np.abs(candidates[ties] - nominal))]
        command = float(candidates[chosen])
    residual = float(np.min(b * command + c)) if b.size else float("inf")
    return IntervalResult(command, lo, hi, bool(feasible), residual)


def pairwise_affine(q, gradient, ego_heading, ego_speed, other_speed,
                    turn_bound, disturbance_bound):
    """Return b,a such that worst-case dot(V) = b*omega_ego+a.

    Each robot has independent WORLD-axis velocity errors in [-D,D]^2.
    Their relative error has support 2*D*||R(theta_ego)*gradient_xy||_1.
    """
    q, gradient = np.asarray(q), np.asarray(gradient)
    x, y, phi = np.moveaxis(q, -1, 0)
    px, py, pt = np.moveaxis(gradient, -1, 0)
    b = px * y - py * x - pt
    cosine, sine = np.cos(ego_heading), np.sin(ego_heading)
    world_px, world_py = cosine * px - sine * py, sine * px + cosine * py
    a = (px * (other_speed * np.cos(phi) - ego_speed)
         + py * other_speed * np.sin(phi) - turn_bound * np.abs(pt)
         - 2 * disturbance_bound * (np.abs(world_px) + np.abs(world_py)))
    return b, a


@dataclass
class ShieldDecision:
    turns: np.ndarray
    interval_feasible: np.ndarray
    all_constraints_satisfied: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    active_pairs: np.ndarray
    min_residual: np.ndarray
    min_barrier: np.ndarray
    outside_domain: np.ndarray
    far_pairs: int

    @property
    def needs_replan(self):
        return (~self.all_constraints_satisfied | (self.min_barrier < 0)
                | self.outside_domain)


class InfeasibleSafetyConstraints(RuntimeError):
    def __init__(self, decision):
        self.decision = decision
        super().__init__("No locally admissible shared command or state outside barrier set; "
                         "a verified fallback/replan is required")


class CompatibleShield:
    """All-threat interval projection, with an explicitly requested fallback.

    `fallback='raise'` prevents silently running an infeasible controller.
    Research rollouts may request 'least_violation' and record the failures.
    `mode='most_critical'` is a one-threat ablation, NOT Chen's MIP algorithm.
    """
    def __init__(self, table: ReachabilityTable, *, gain=2.0, rate_margin=0.0,
                 disturbance_bound=0.12, fallback="raise", mode="all"):
        if (not np.isfinite(gain) or gain <= 0 or not np.isfinite(rate_margin)
                or rate_margin < 0 or not np.isfinite(disturbance_bound)
                or disturbance_bound < 0):
            raise ValueError("Invalid barrier gain, residual margin, or disturbance bound")
        if fallback not in ("raise", "least_violation") or mode not in ("all", "most_critical"):
            raise ValueError("Unknown fallback or constraint mode")
        self.table, self.gain, self.rate_margin = table, gain, rate_margin
        self.disturbance_bound, self.fallback, self.mode = disturbance_bound, fallback, mode
        self.turn_bound = float(table.meta["turn_rate"])

    def filter(self, poses, nominal_turns, speeds, *, dt):
        poses, nominal, speeds = (np.asarray(x, dtype=float)
                                  for x in (poses, nominal_turns, speeds))
        n = len(poses)
        if (n < 1 or poses.shape != (n, 3) or nominal.shape != (n,) or speeds.shape != (n,)
                or not all(np.isfinite(x).all() for x in (poses, nominal, speeds))
                or np.any(speeds < 0) or np.any(speeds > self.table.meta["speed"] + 1e-12)
                or not np.isfinite(dt) or dt <= 0):
            raise ValueError("Invalid poses, nominal commands, speeds, or timestep")
        q = relative_state(poses[:, None, :], poses[None, :, :])
        in_domain = self.table.contains(q)
        np.fill_diagonal(in_domain, False)
        ego, other = np.where(in_domain)
        values, gradients = self.table.value_and_gradient(q[ego, other])
        barriers = values - self.table.geometric_padding - self.table.margin
        b, a = pairwise_affine(q[ego, other], gradients, poses[ego, 2],
                              speeds[ego], speeds[other], self.turn_bound,
                              self.disturbance_bound)
        c = a + self.gain * barriers - self.rate_margin
        rows = [[] for _ in range(n)]
        min_h = np.full(n, np.inf)
        for row, (i, j) in enumerate(zip(ego, other)):
            min_h[i] = min(min_h[i], barriers[row])
            # Branches are needed only on grid faces (including the theta seam).
            pos = (q[i, j] - np.array([axis[0] for axis in self.table.axes])) / self.table.spacing
            face = np.any(np.isclose(pos, np.round(pos), rtol=0, atol=1e-9))
            if face:
                branches = self.table.gradient_branches(q[i, j])
                bb, aa = pairwise_affine(q[i, j], branches, poses[i, 2],
                                        speeds[i], speeds[j], self.turn_bound,
                                        self.disturbance_bound)
                cc = aa + self.gain * barriers[row] - self.rate_margin
                rows[i].extend((j, float(bi), float(ci), barriers[row]) for bi, ci in zip(bb, cc))
            else:
                rows[i].append((j, b[row], c[row], barriers[row]))
        # No clamping of near out-of-domain states. Distant pairs have a
        # geometric no-contact bound through at least the next control sample.
        distance = np.linalg.norm(q[..., :2], axis=-1)
        closing = 2 * (float(self.table.meta["speed"]) + np.sqrt(2) * self.disturbance_bound)
        far = distance > self.table.capture_radius + closing * max(dt, self.table.meta["horizon"])
        missing = ~in_domain & ~np.eye(n, dtype=bool)
        unsupported = np.any(missing & ~far, axis=1)
        decisions, active, satisfied = [], [], []
        for i, entries in enumerate(rows):
            if entries:
                array = np.asarray(entries)
                # Count opponents that restrict at least one bounded turn rate.
                relevant = array[:, 2] - np.abs(array[:, 1]) * self.turn_bound < 0
                active.append(len(np.unique(array[relevant, 0])))
                selected = array
                if self.mode == "most_critical":
                    threat = min(entries, key=lambda row: (row[3], row[0]))[0]
                    selected = array[array[:, 0] == threat]
                decision = project_interval(nominal[i], selected[:, 1], selected[:, 2], self.turn_bound)
                residual = float(np.min(array[:, 1] * decision.control + array[:, 2]))
                decision.min_residual = residual
            else:
                active.append(0)
                decision = project_interval(nominal[i], [], [], self.turn_bound)
            if unsupported[i]:
                decision.feasible = False
            satisfied.append(decision.min_residual >= -1e-9 and not unsupported[i])
            decisions.append(decision)
        result = ShieldDecision(
            turns=np.array([r.control for r in decisions]),
            interval_feasible=np.array([r.feasible for r in decisions]),
            all_constraints_satisfied=np.array(satisfied),
            lower=np.array([r.lower for r in decisions]),
            upper=np.array([r.upper for r in decisions]),
            active_pairs=np.array(active), min_residual=np.array([r.min_residual for r in decisions]),
            min_barrier=min_h, outside_domain=unsupported,
            far_pairs=int(np.count_nonzero(missing & far)))
        if self.fallback == "raise" and np.any(result.needs_replan):
            raise InfeasibleSafetyConstraints(result)
        return result
