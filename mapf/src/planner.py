"""Windowed prioritized planning with a pluggable conflict predicate.

RHCR-style rolling horizon (innovation #1): at each replanning tick we plan a
timed path over a bounded window `w` and only resolve conflicts within it.
Agents are planned in priority order; each commits a timed path that lower-
priority agents must avoid (prioritized planning; the low-level of CBS/RHCR).

The conflict predicate is the pluggable piece (innovation #2):
  * GeometricPredicate -- classical dynamics-blind test: vertex (same cell) and
    edge-swap conflicts only. Adjacency is allowed.
  * HJPredicate        -- vertex conflicts PLUS the dynamics-aware BRT test: a
    candidate move is blocked if the continuous relative Dubins state w.r.t. a
    higher-priority agent lies inside the precomputed windowed BRT.

Both share the same space-time A* so the ONLY difference between policies is the
predicate — a clean, CRN-friendly contrast.
"""

from __future__ import annotations
import heapq
import numpy as np

from mapf_world import WarehouseGrid, MOVES


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _heading(from_c, to_c, default):
    dr, dc = to_c[0] - from_c[0], to_c[1] - from_c[1]
    if dr == 0 and dc == 0:
        return default
    return float(np.arctan2(dr, dc))  # y=row, x=col -> atan2(dy,dx)


class GeometricPredicate:
    """Dynamics-blind: vertex + edge-swap conflicts only (classical MAPF/RHCR)."""
    name = "geometric"

    def blocked(self, from_c, to_c, t, committed):
        for path in committed:
            cells = path["cells"]
            c_next = cells[min(t + 1, len(cells) - 1)]
            c_cur = cells[min(t, len(cells) - 1)]
            if to_c == c_next:            # vertex conflict
                return True
            if to_c == c_cur and from_c == c_next:   # edge swap
                return True
        return False


class HJPredicate:
    """Dynamics-aware: vertex conflicts + windowed BRT membership."""
    name = "hj"

    def __init__(self, brt, grid: WarehouseGrid, self_default_theta=0.0):
        self.brt = brt
        self.grid = grid
        self.default_theta = self_default_theta

    def blocked(self, from_c, to_c, t, committed):
        # exact overlap is always a conflict
        for path in committed:
            cells = path["cells"]
            if to_c == cells[min(t + 1, len(cells) - 1)]:
                return True
        # dynamics-aware BRT test on continuous poses
        th = _heading(from_c, to_c, self.default_theta)
        self_pose = np.array([float(to_c[1]), float(to_c[0]), th])
        for path in committed:
            poses = path["poses"]
            other = poses[min(t + 1, len(poses) - 1)]
            # symmetric: either agent as pursuer
            if self.brt.in_conflict(self_pose, other) or \
               self.brt.in_conflict(other, self_pose):
                return True
        return False


def _astar_windowed(grid, start, goal, predicate, committed, window,
                    start_theta, max_expansions=20000):
    """Space-time A* to `goal`; conflicts checked only for t < window.

    Returns (cells, poses) timed path (index = t) of length >= 1. `poses` are
    (x, y, theta) at each t. Falls back to a wait-in-place step if boxed in.
    """
    # node = (cell, t); g-cost = t (steps incl. waits); h = manhattan
    max_t = window + 2 * (grid.H + grid.W)
    start_node = (start, 0)
    openq = [(_manhattan(start, goal), 0, start, 0, start_theta)]
    came = {}                      # (cell,t) -> (prev_cell, prev_t, theta)
    best_g = {start_node: 0}
    expansions = 0
    goal_node = None

    while openq and expansions < max_expansions:
        f, g, cell, t, th = heapq.heappop(openq)
        expansions += 1
        if cell == goal:
            goal_node = (cell, t, th)
            break
        if t >= max_t:
            goal_node = (cell, t, th)   # best effort
            break
        for dr, dc in MOVES:
            nb = (cell[0] + dr, cell[1] + dc)
            if not grid.is_free(nb):
                continue
            # only enforce conflicts inside the rolling window
            if t < window and predicate.blocked(cell, nb, t, committed):
                continue
            ng = g + 1
            nnode = (nb, t + 1)
            if ng < best_g.get(nnode, 1 << 30):
                best_g[nnode] = ng
                nth = _heading(cell, nb, th)
                came[nnode] = (cell, t, nth)
                heapq.heappush(openq, (ng + _manhattan(nb, goal), ng, nb, t + 1, nth))

    if goal_node is None:
        # boxed in: wait in place
        return [start, start], [
            np.array([float(start[1]), float(start[0]), start_theta])
        ] * 2

    # reconstruct
    cells, thetas = [], []
    node = (goal_node[0], goal_node[1])
    th = goal_node[2]
    while node != start_node:
        cells.append(node[0])
        thetas.append(th)
        prev_c, prev_t, prev_th = came[node]
        node = (prev_c, prev_t)
        th = prev_th
    cells.append(start)
    thetas.append(start_theta)
    cells.reverse()
    thetas.reverse()
    poses = [np.array([float(c[1]), float(c[0]), thetas[i]]) for i, c in enumerate(cells)]
    return cells, poses


def plan_window(grid, agents, predicate, window):
    """Prioritized windowed planning. Returns per-agent (cells, poses) + stats.

    Agents planned in ascending `priority` (0 = highest). Each commits its timed
    path as a reservation the next agents must respect via `predicate`.
    """
    order = sorted(agents, key=lambda a: a.priority)
    committed = []
    plans = {}
    total_wait = 0
    for a in order:
        cells, poses = _astar_windowed(
            grid, a.cell, a.goal, predicate, committed, window, a.theta
        )
        # count waits within the window (conservatism proxy)
        for t in range(1, min(len(cells), window + 1)):
            if cells[t] == cells[t - 1]:
                total_wait += 1
        committed.append({"cells": cells, "poses": poses})
        plans[a.id] = (cells, poses)
    return plans, {"wait_steps": total_wait}
