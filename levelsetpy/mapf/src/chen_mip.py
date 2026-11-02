"""Chen et al. (CDC 2016), equations (4)--(7), with a matched finite-horizon table.

The assignment MIP is reproduced; the table, speeds, disturbances, and nominal
controller are those of this repository. This is NOT the paper's infinite-
horizon safety guarantee. N=3 uses its published priority matrix; larger N
uses a documented cyclic ordering extension. K defaults to 0.15 (K/r_c=0.3,
the ratio in the original experiment). Both K=0.1 and K=0.15 are evaluated.
"""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from compatible_shield import CompatibleShield, pairwise_affine, project_interval
from hj_conflict import relative_state


def priority_matrix(n):
    if n == 3:
        return np.array([[0, 6, 3], [2, 0, 5], [4, 1, 0]])
    p = np.zeros((n, n), dtype=int)
    rank = n * (n - 1)
    for offset in range(1, n):
        for i in range(n):
            p[i, (i + offset) % n] = rank
            rank -= 1
    return p


def solve_assignment(rewards):
    n = len(rewards)
    edges = [(i, j) for i in range(n) for j in range(n) if i != j]
    if not edges:
        return np.zeros((n, n), dtype=int)
    constraints = []
    for i in range(n):
        constraints.append([int(a == i) for a, b in edges])
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append([int({a, b} == {i, j}) for a, b in edges])
    result = milp(-np.array([rewards[i, j] for i, j in edges], dtype=float),
                  integrality=np.ones(len(edges)), bounds=Bounds(0, 1),
                  constraints=LinearConstraint(np.array(constraints), -np.inf, 1),
                  options={"mip_rel_gap": 0.0})
    if not result.success:
        raise RuntimeError(f"Assignment solver failed: {result.message}")
    assignment = np.zeros((n, n), dtype=int)
    for (i, j), value in zip(edges, np.rint(result.x).astype(int)):
        assignment[i, j] = value
    if np.any(assignment.sum(axis=1) > 1) or np.any(assignment + assignment.T > 1):
        raise AssertionError("Invalid binary responsibility assignment")
    return assignment


class ChenMIPShield(CompatibleShield):
    def __init__(self, table, *, threshold=.15, **kwargs):
        super().__init__(table, **kwargs)
        self.threshold = threshold
        self.assignment_cache = {}
        self.mip_solves = 0
        self.cache_hits = 0

    def filter(self, poses, nominal_turns, speeds, *, dt):
        # Evaluate common-barrier diagnostics using the same implementation as
        # the simultaneous arm, then replace its command by MIP/HJ commands.
        result = super().filter(poses, nominal_turns, speeds, dt=dt)
        poses, speeds = np.asarray(poses), np.asarray(speeds)
        n = len(poses)
        q = relative_state(poses[:, None, :], poses[None, :, :])
        inside = self.table.contains(q)
        np.fill_diagonal(inside, False)
        ii, jj = np.where(inside)
        values, gradients = self.table.value_and_gradient(q[ii, jj])
        potential = np.zeros((n, n), dtype=bool)
        potential[ii, jj] = values <= self.threshold
        key = (n, potential.tobytes())
        if key not in self.assignment_cache:
            rewards = np.where(potential, priority_matrix(n) ** 2, -1)
            self.assignment_cache[key] = solve_assignment(rewards)
            self.mip_solves += 1
        else:
            self.cache_hits += 1
        assignment = self.assignment_cache[key]
        turns = np.array(nominal_turns, dtype=float, copy=True)
        rows = [[] for _ in range(n)]
        for row, (i, j) in enumerate(zip(ii, jj)):
            branches = self.table.gradient_branches(q[i, j])
            b, a = pairwise_affine(q[i, j], branches, poses[i, 2],
                                  speeds[i], speeds[j], self.turn_bound,
                                  self.disturbance_bound)
            h = values[row] - self.table.geometric_padding - self.table.margin
            rows[i].extend(zip(b, a + self.gain * h - self.rate_margin))
            if assignment[i, j]:
                # Maximize worst-case derivative of the pairwise value. For
                # one gradient this is the HJ bang-bang control. At a cell
                # face maximize the minimum over incident gradient branches.
                # Force the max-min branch of the interval helper by shifting
                # every offset below its maximum attainable value.
                shift = float(np.max(a + np.abs(b) * self.turn_bound)) + 1
                turns[i] = project_interval(turns[i], b, a - shift,
                                             self.turn_bound).control
        result.turns = turns
        for i, entries in enumerate(rows):
            result.min_residual[i] = min((b * turns[i] + c for b, c in entries), default=np.inf)
        result.all_constraints_satisfied = ((result.min_residual >= -1e-9)
                                           & ~result.outside_domain)
        self.last_assignment = assignment.copy()
        return result
