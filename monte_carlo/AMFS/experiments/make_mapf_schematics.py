#!/usr/bin/env python
"""Schematic figures introducing MAPF on a grid-partitioned warehouse floor.

Map family follows the RHCR fulfillment-warehouse convention (Li et al., AAAI
2021): rows of storage blocks separated by single-cell travel corridors, with
task endpoints on the left/right border columns.

  floor_cbs.png   -- the discrete abstraction CBS/ECBS search over, with the
                     two classical conflict types marked (vertex, edge/swap)
  floor_ccbs.png  -- what CCBS adds (continuous time, disc geometry, unsafe
                     intervals) and what it still leaves out

Numpy + matplotlib only.
"""
from __future__ import annotations
import os
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch

INK = "#0a1730"
BLOCK = "#c9cfe3"
BLOCK_E = "#98a1c3"
A1 = "#1f4fd8"
A2 = "#c0392b"
GOLD = "#c47a00"
TEAL = "#0c7e80"


def build_floor(block_rows=3, block_cols=4, block_h=2, block_w=3,
                corridor=1, border=2):
    H = 2 * border + block_rows * block_h + (block_rows - 1) * corridor
    W = 2 * border + block_cols * block_w + (block_cols - 1) * corridor
    obs = np.zeros((H, W), bool)
    for br in range(block_rows):
        r0 = border + br * (block_h + corridor)
        for bc in range(block_cols):
            c0 = border + bc * (block_w + corridor)
            obs[r0:r0 + block_h, c0:c0 + block_w] = True
    endpoints = [(r, c) for r in range(border, H - border) for c in (0, W - 1)
                 if not obs[r, c]]
    return obs, endpoints


def draw_grid(ax, obs, endpoints, lw=0.6):
    H, W = obs.shape
    for r in range(H + 1):
        ax.plot([-0.5, W - 0.5], [r - 0.5, r - 0.5], color="#c8cde0", lw=lw, zorder=1)
    for c in range(W + 1):
        ax.plot([c - 0.5, c - 0.5], [-0.5, H - 0.5], color="#c8cde0", lw=lw, zorder=1)
    for (r, c) in np.argwhere(obs):
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=BLOCK,
                               edgecolor=BLOCK_E, lw=0.7, zorder=2))
    for (r, c) in endpoints:
        ax.add_patch(Rectangle((c - 0.35, r - 0.35), 0.7, 0.7, facecolor="#ffe3b3",
                               edgecolor=GOLD, lw=1.1, zorder=3))
    ax.set_xlim(-1.2, W + 0.2)
    ax.set_ylim(H - 0.5, -1.6)
    ax.set_aspect("equal")
    ax.axis("off")


def path_line(ax, cells, color, label=None, ls="-", lw=2.6, ms=0):
    xs = [c for (_, c) in cells]
    ys = [r for (r, _) in cells]
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, solid_capstyle="round",
            zorder=6, label=label, marker="o", ms=ms)


def fig_cbs(out_dir):
    obs, endpoints = build_floor()
    H, W = obs.shape
    fig, ax = plt.subplots(figsize=(11.8, 6.2), dpi=170)
    draw_grid(ax, obs, endpoints)

    # Row 4 and column 5 are travel corridors; every cell used below is free.
    p1 = [(4, c) for c in range(0, 9)]            # a1: left border -> east
    p2 = [(r, 5) for r in range(1, 8)]            # a2: north -> south
    p3 = [(4, c) for c in range(12, 5, -1)]       # a3: east -> west (head-on a1)
    for cells, col, ls in ((p1, A1, "-"), (p2, A2, "--"), (p3, TEAL, "-.")):
        for (r, c) in cells:
            assert not obs[r, c], f"path crosses a block at {(r, c)}"
        path_line(ax, cells, col, ls=ls, ms=4.2)

    for (r, c), col, name, dx, dy in (((4, 0), A1, "$a_1$", -1.05, 0),
                                      ((1, 5), A2, "$a_2$", 0, -0.95),
                                      ((4, 12), TEAL, "$a_3$", 0, -0.95)):
        ax.add_patch(Circle((c, r), 0.34, facecolor=col, edgecolor="white",
                            lw=1.4, zorder=8))
        ax.text(c + dx, r + dy, name, color=col, fontsize=15, fontweight="bold",
                ha="center", va="center", zorder=8)
    for (r, c), col in (((4, 8), A1), ((7, 5), A2), ((4, 6), TEAL)):
        ax.plot(c, r, marker="*", ms=16, color=col, zorder=8,
                markeredgecolor="white", markeredgewidth=0.8)

    # vertex conflict: a1 and a2 both want the corridor junction (4,5)
    ax.add_patch(Circle((5, 4), 0.60, facecolor="none", edgecolor=GOLD, lw=3.0,
                        zorder=9))
    ax.annotate("vertex conflict\n$\\langle a_1,a_2,(4,5),t\\rangle$",
                xy=(5, 4.45), xytext=(2.6, 11.0), fontsize=13.5,
                fontweight="bold", color=GOLD, ha="center",
                arrowprops=dict(arrowstyle="->", color=GOLD, lw=2.0))

    # edge (swap) conflict: a1 and a3 traverse the same edge in opposition
    ax.add_patch(FancyArrowPatch((7.05, 3.55), (7.95, 3.55), arrowstyle="<->",
                                 mutation_scale=15, color="#8a3b00", lw=2.6,
                                 zorder=9))
    ax.annotate("edge (swap) conflict\n$\\langle a_1,a_3,(4,7)\\!\\to\\!(4,8),t\\rangle$",
                xy=(7.5, 3.5), xytext=(11.4, 11.0), fontsize=13.5,
                fontweight="bold", color="#8a3b00", ha="center",
                arrowprops=dict(arrowstyle="->", color="#8a3b00", lw=2.0))

    ax.text(-0.4, -1.30, "storage block", fontsize=12.5, color=BLOCK_E,
            fontweight="bold", va="center")
    ax.add_patch(Rectangle((-1.15, -1.55), 0.55, 0.55, facecolor=BLOCK,
                           edgecolor=BLOCK_E, lw=0.9, clip_on=False, zorder=5))
    ax.text(5.6, -1.30, "task endpoint", fontsize=12.5, color=GOLD,
            fontweight="bold", va="center")
    ax.add_patch(Rectangle((4.35, -1.55), 0.55, 0.55, facecolor="#ffe3b3",
                           edgecolor=GOLD, lw=1.1, clip_on=False, zorder=5))
    ax.text(11.2, -1.30, "single-cell travel corridor", fontsize=12.5, color=INK,
            fontweight="bold", va="center")

    ax.set_title("Discrete MAPF: the grid-partitioned floor that CBS / ECBS / RHCR "
                 "search over", fontsize=16.5, fontweight="bold", color=INK,
                 pad=14)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.02)
    p = os.path.join(out_dir, "floor_cbs.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def fig_ccbs(out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.3), dpi=170,
                             gridspec_kw=dict(width_ratios=[1.05, 1.0], wspace=0.16))

    # ---------- left: disc agents on a 2^k neighbourhood, continuous time -----
    ax = axes[0]
    for r in range(5):
        ax.plot([-0.5, 4.5], [r - 0.5, r - 0.5], color="#c8cde0", lw=0.7, zorder=1)
    for c in range(6):
        ax.plot([c - 0.5, c - 0.5], [-0.5, 3.5], color="#c8cde0", lw=0.7, zorder=1)
    rad = 0.42
    # a1 moves along a diagonal (2^k) edge; a2 along a shallower one; they meet
    ax.plot([0, 3], [0, 3], color=A1, lw=2.6, zorder=5)
    ax.plot([4, 0], [0, 2], color=A2, lw=2.6, ls="--", zorder=5)
    meet = (1.6, 1.6)
    for (x, y) in ((0, 0), meet):
        ax.add_patch(Circle((x, y), rad, facecolor=A1, alpha=0.9,
                            edgecolor="white", lw=1.2, zorder=6))
    for (x, y) in ((4, 0), (1.15, 1.42)):
        ax.add_patch(Circle((x, y), rad, facecolor=A2, alpha=0.9,
                            edgecolor="white", lw=1.2, zorder=6))
    ax.add_patch(Circle(meet, rad * 1.9, facecolor="none", edgecolor=GOLD,
                        lw=2.4, zorder=7))
    ax.annotate("discs overlap:\ncollision at a non-integer time",
                xy=(1.9, 1.9), xytext=(2.9, 3.05), fontsize=12.5,
                fontweight="bold", color=GOLD, ha="center",
                arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.9))
    ax.text(-0.4, -1.0, "straight-line moves, arbitrary durations, "
                        "$2^k$-neighbourhood", fontsize=12, color=INK,
            fontweight="bold")
    ax.set_xlim(-1.0, 5.0); ax.set_ylim(3.6, -1.4)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("CCBS: geometric agents in continuous time",
                 fontsize=15, fontweight="bold", color=INK, pad=10)

    # ---------- right: unsafe interval timeline -------------------------------
    ax = axes[1]
    ax.plot([0, 10], [1, 1], color=INK, lw=2.2, zorder=3)
    ax.plot([0, 10], [0, 0], color=INK, lw=2.2, zorder=3)
    ax.add_patch(Rectangle((3.1, 0.86), 2.6, 0.28, facecolor="#f3c9c9",
                           edgecolor=A2, lw=1.6, zorder=4))
    ax.add_patch(Rectangle((4.0, -0.14), 2.4, 0.28, facecolor="#f3c9c9",
                           edgecolor=A2, lw=1.6, zorder=4))
    ax.text(4.4, 1.42, "unsafe interval for $a_1$", fontsize=12.5,
            fontweight="bold", color=A2, ha="center")
    ax.text(5.2, -0.52, "unsafe interval for $a_2$", fontsize=12.5,
            fontweight="bold", color=A2, ha="center")
    ax.text(-0.15, 1.0, "$a_1$", fontsize=14, fontweight="bold", color=A1,
            ha="right", va="center")
    ax.text(-0.15, 0.0, "$a_2$", fontsize=14, fontweight="bold", color=A2,
            ha="right", va="center")
    ax.annotate("", xy=(10.15, 0.5), xytext=(9.4, 0.5),
                arrowprops=dict(arrowstyle="->", color=INK, lw=2.0))
    ax.text(10.2, 0.5, "time", fontsize=12.5, fontweight="bold", color=INK,
            va="center")
    ax.text(5.0, -1.15,
            "Constraints become time intervals, not integer timesteps.\n"
            "Still absent: turn radius, actuation lag, localization noise.",
            fontsize=12.5, fontweight="bold", color=INK, ha="center")
    ax.set_xlim(-1.3, 11.6); ax.set_ylim(-1.7, 1.95)
    ax.axis("off")
    ax.set_title("Conflicts as unsafe time intervals",
                 fontsize=15, fontweight="bold", color=INK, pad=10)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.04)
    p = os.path.join(out_dir, "floor_ccbs.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    fig_cbs(a.out_dir)
    fig_ccbs(a.out_dir)
