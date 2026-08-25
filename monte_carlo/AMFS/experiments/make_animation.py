#!/usr/bin/env python
"""Animate one CRN-paired AMFS episode: geometric vs HJ-shield, side by side.

Runs the SAME episode (same seed, same spawn/task/disturbance streams) under
both policies, traces the true continuous executor micro-steps, and renders:

  amfs_rollout.gif        -- animated side-by-side comparison (for HTML slides)
  amfs_rollout_strip.png  -- 4-frame static filmstrip (for PDF slides)

Run under the AMFS venv:
    ./.venv/bin/python experiments/make_animation.py --out-dir <slides assets>
"""
from __future__ import annotations
import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
for p in (_HERE, _SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from exp_A_pipeline import run_episode, make_grid, POLICY_GEOMETRIC, POLICY_HJ

TEAL = "#0c7e80"
RED = "#c0392b"
AMBER = "#e08a00"
INK = "#0a1730"


def draw_floor(ax, grid):
    for (r, c) in np.argwhere(grid.obstacle):
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor="#c9cfe3",
                               edgecolor="#9aa3c4", linewidth=0.4, zorder=1))
    ax.set_xlim(-1.0, grid.W)
    ax.set_ylim(grid.H, -1.0)          # row 0 on top
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#9aa3c4")


def draw_frame(ax, grid, fr, radius, title, subtitle_color):
    draw_floor(ax, grid)
    in_contact = set()
    for (i, j) in fr["contacts"]:
        in_contact.add(i); in_contact.add(j)

    for aid, (cell) in fr["goals"].items():
        ax.plot(cell[1], cell[0], marker="*", ms=7, color="#8a8fb0",
                zorder=2, linestyle="none")

    for aid, pose in fr["poses"].items():
        x, y, th = pose
        yielding = aid in fr["yielding"]
        hit = aid in in_contact
        fc = RED if hit else (AMBER if yielding else TEAL)
        ax.add_patch(Circle((x, y), radius, facecolor=fc, edgecolor="white",
                            linewidth=0.7, zorder=4))
        ax.arrow(x, y, 0.55 * np.cos(th), 0.55 * np.sin(th),
                 head_width=0.22, head_length=0.2, fc=INK, ec=INK,
                 linewidth=0.5, zorder=5, length_includes_head=True)
        if yielding:
            ax.add_patch(Circle((x, y), radius * 2.6, facecolor="none",
                                edgecolor=AMBER, linewidth=1.2, ls="--", zorder=3))

    ax.set_title(title, fontsize=11, fontweight="bold", color=subtitle_color,
                 pad=6)
    ax.text(0.5, -0.045,
            f"realized collisions: {fr['collisions']:.0f}"
            + (f"   ·   shield fired: {fr['interventions']:.0f}"
               if fr["interventions"] or "HJ" in title else ""),
            transform=ax.transAxes, ha="center", va="top",
            fontsize=10, fontweight="bold", color=INK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=1028,
                    help="1028 reproduces the aggregate effect (98 -> 66)")
    ap.add_argument("--n-agents", type=int, default=14)
    ap.add_argument("--total-ticks", type=int, default=40)
    ap.add_argument("--stride", type=int, default=5, help="keep every Nth micro-step")
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--dpi", type=int, default=76)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    grid = make_grid()
    traces, results = {}, {}
    for policy in (POLICY_GEOMETRIC, POLICY_HJ):
        tr = []
        res = run_episode(args.seed, policy, n_agents=args.n_agents,
                          total_ticks=args.total_ticks, grid=make_grid(), trace=tr)
        traces[policy], results[policy] = tr, res
        print(f"  {policy:<10} collisions={res['collisions']:.0f} "
              f"shield={res['interventions']:.0f} frames={len(tr)}")

    n = min(len(traces[POLICY_GEOMETRIC]), len(traces[POLICY_HJ]))
    idx = list(range(0, n, args.stride))
    radius = 0.25

    frames = []
    for k, i in enumerate(idx):
        fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6), dpi=args.dpi)
        draw_frame(axes[0], grid, traces[POLICY_GEOMETRIC][i], radius,
                   "Geometric conflict check (dynamics-blind)", RED)
        draw_frame(axes[1], grid, traces[POLICY_HJ][i], radius,
                   "HJ-Gauss windowed-BRT shield (certified)", TEAL)
        fig.suptitle(f"AMFS structured floor · {args.n_agents} lifelong agents · "
                     f"same seed, same disturbance stream   (t = {i * 0.1:.1f} s)",
                     fontsize=12, fontweight="bold", color=INK, y=0.99)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.07, wspace=0.05)
        fig.canvas.draw()
        rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        frames.append(Image.fromarray(rgb).convert(
            "P", palette=Image.ADAPTIVE, colors=128))
        plt.close(fig)
        if k % 25 == 0:
            print(f"    frame {k}/{len(idx)}")

    gif = os.path.join(args.out_dir, "amfs_rollout.gif")
    frames[0].save(gif, save_all=True, append_images=frames[1:], loop=0,
                   duration=int(1000 / args.fps), optimize=True)
    print(f"wrote {gif}  ({len(frames)} frames, "
          f"{os.path.getsize(gif) / 1e6:.1f} MB)")

    # static filmstrip for the PDF: 4 evenly spaced frames of the HJ/geom pair
    picks = np.linspace(0, len(idx) - 1, 4).round().astype(int)
    fig, axes = plt.subplots(2, 4, figsize=(13.6, 6.4), dpi=110)
    for col, pk in enumerate(picks):
        i = idx[pk]
        draw_frame(axes[0, col], grid, traces[POLICY_GEOMETRIC][i], radius,
                   f"Geometric · t={i * 0.1:.1f} s", RED)
        draw_frame(axes[1, col], grid, traces[POLICY_HJ][i], radius,
                   f"HJ-shield · t={i * 0.1:.1f} s", TEAL)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.04,
                        wspace=0.06, hspace=0.22)
    strip = os.path.join(args.out_dir, "amfs_rollout_strip.png")
    fig.savefig(strip, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {strip}")

    print("\nEpisode totals (this seed):")
    for policy in (POLICY_GEOMETRIC, POLICY_HJ):
        r = results[policy]
        print(f"  {policy:<10} collisions={r['collisions']:.0f} "
              f"throughput={r['throughput']:.4f} shield={r['interventions']:.0f}")


if __name__ == "__main__":
    main()
