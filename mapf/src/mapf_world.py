"""RHCR-style structured warehouse grid + lifelong task assignment (numpy only).

Mirrors the map family of Li et al. (RHCR, AAAI 2021): rows of storage blocks
(obstacles) separated by single-cell travel corridors, with endpoints (agent
home / task locations) on the left and right borders. Lifelong: whenever an
agent reaches its goal it is immediately assigned a new random endpoint, so the
system runs continuously and we measure throughput (goals reached per step).
"""

from __future__ import annotations
import numpy as np

# 4-connected moves + wait. (dr, dc)
MOVES = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]


class WarehouseGrid:
    def __init__(self, block_rows=3, block_cols=4, block_h=2, block_w=3,
                 corridor=1, border=2):
        """Build a structured floor.

        block_rows x block_cols storage blocks, each block_h x block_w cells,
        separated by `corridor`-wide free lanes, with a `border`-wide free ring
        whose left/right columns hold the endpoints.
        """
        self.block_rows = block_rows
        self.block_cols = block_cols
        self.block_h = block_h
        self.block_w = block_w
        self.corridor = corridor
        self.border = border

        H = 2 * border + block_rows * block_h + (block_rows - 1) * corridor
        W = 2 * border + block_cols * block_w + (block_cols - 1) * corridor
        self.H, self.W = H, W
        self.obstacle = np.zeros((H, W), dtype=bool)

        for br in range(block_rows):
            r0 = border + br * (block_h + corridor)
            for bc in range(block_cols):
                c0 = border + bc * (block_w + corridor)
                self.obstacle[r0:r0 + block_h, c0:c0 + block_w] = True

        # endpoints: free cells on the left and right border columns
        self.endpoints = []
        for r in range(border, H - border):
            for c in (0, W - 1):
                if not self.obstacle[r, c]:
                    self.endpoints.append((r, c))
        self.free_cells = [tuple(x) for x in np.argwhere(~self.obstacle)]

    def is_free(self, cell):
        r, c = cell
        return 0 <= r < self.H and 0 <= c < self.W and not self.obstacle[r, c]

    def neighbors(self, cell):
        out = []
        for dr, dc in MOVES:
            nb = (cell[0] + dr, cell[1] + dc)
            if self.is_free(nb):
                out.append(nb)
        return out

    def cell_to_xy(self, cell):
        """Map (row, col) -> continuous (x, y) at cell center (x=col, y=row)."""
        return np.array([float(cell[1]), float(cell[0])])


class Agent:
    __slots__ = ("id", "cell", "theta", "goal", "priority", "goals_reached")

    def __init__(self, aid, cell, goal, theta=0.0, priority=0):
        self.id = aid
        self.cell = cell
        self.theta = float(theta)
        self.goal = goal
        self.priority = priority
        self.goals_reached = 0


class LifelongTasks:
    """Assigns random endpoint goals; reassigns on arrival."""

    def __init__(self, grid: WarehouseGrid, rng: np.random.Generator):
        self.grid = grid
        self.rng = rng

    def new_goal(self, exclude=None):
        eps = self.grid.endpoints
        while True:
            g = eps[self.rng.integers(len(eps))]
            if g != exclude:
                return g


def spawn_agents(grid: WarehouseGrid, n_agents: int, rng: np.random.Generator):
    """Place agents at distinct free cells with distinct initial goals."""
    free = [c for c in grid.free_cells]
    rng.shuffle(free)
    starts = free[:n_agents]
    tasks = LifelongTasks(grid, rng)
    agents = []
    for i, s in enumerate(starts):
        g = tasks.new_goal(exclude=s)
        agents.append(Agent(i, s, g, theta=0.0, priority=i))
    return agents, tasks
