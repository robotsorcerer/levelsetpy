#!/usr/bin/env python
"""Animate the cached pairwise Dubins conflict set across relative headings.

Loads the cached windowed BRT (cache/dubins_brt.npz, solved by HJ-Gauss
Algorithm 1 at the rolling-window horizon) and sweeps the relative-heading
coordinate x3, overlaying the dynamics-blind capture disc. This is the
heading dependence a geometric conflict check cannot represent.

Outputs (into --out-dir):
  brt_heading.gif        -- animated sweep, for the HTML deck
  brt_heading_strip.png  -- 4-panel static filmstrip, for the PDF deck

Numpy only (no JAX): reads the cache written by precompute_brt.py.
"""
from __future__ import annotations
import os
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image

INK = "#0a1730"
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BRT = os.path.join(os.path.dirname(_HERE), "cache", "dubins_brt.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--brt", default=DEFAULT_BRT)
    ap.add_argument("--fps", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    d = np.load(args.brt, allow_pickle=True)
    V = np.asarray(d["V"], dtype=float)                  # (n_xy, n_xy, n_theta)
    x1, x2, th = (np.asarray(d[k], dtype=float)
                  for k in ("x1_axis", "x2_axis", "th_axis"))
    try:
        meta = eval(str(d["meta"][0]), {"__builtins__": {}})
    except Exception:
        meta = {}
    r_c = float(meta.get("capture_radius", 0.8))
    w = float(meta.get("window_horizon", 0.6))
    X1, X2 = np.meshgrid(x1, x2, indexing="ij")
    lim = float(x1[-1])

    order = np.argsort(th)                               # sweep -pi -> +pi
    print(f"BRT {V.shape}, window w={w}s, r_capture={r_c}, "
          f"frac inside={np.mean(V <= 0):.3f}")

    def draw(ax, k, small=False):
        Vs = V[:, :, k]
        ax.contourf(X1, X2, Vs, levels=[-1e9, 0.0], colors=["#7fb3b4"],
                    alpha=0.9, zorder=2)
        ax.contour(X1, X2, Vs, levels=[0.0], colors=[INK], linewidths=2.2,
                   zorder=3)
        ax.add_patch(Circle((0, 0), r_c, facecolor="none", edgecolor="#c0392b",
                            ls="--", linewidth=1.9, zorder=4))
        ax.plot(0, 0, marker="o", ms=6, color="#c0392b", zorder=5)
        ax.arrow(0, 0, 0.9, 0, head_width=0.16, head_length=0.16, fc="#c0392b",
                 ec="#c0392b", zorder=6, length_includes_head=True)
        a = float(th[k])
        ax.arrow(-lim * 0.72, lim * 0.78, 0.9 * np.cos(a), 0.9 * np.sin(a),
                 head_width=0.16, head_length=0.16, fc="#1f4fd8", ec="#1f4fd8",
                 zorder=6, length_includes_head=True)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        ax.set_xlabel("$x_1$  (along heading)", fontsize=9 if small else 11)
        ax.set_ylabel("$x_2$  (cross heading)", fontsize=9 if small else 11)
        ax.tick_params(labelsize=8 if small else 9)
        ax.grid(alpha=0.25, zorder=0)
        ax.set_title(f"relative heading $x_3={a:+.2f}$ rad  ·  "
                     f"inside: {np.mean(Vs <= 0) * 100:.1f}%",
                     fontsize=10 if small else 12, fontweight="bold", color=INK)

    frames = []
    for k in order:
        fig, ax = plt.subplots(figsize=(5.8, 5.4), dpi=92)
        draw(ax, int(k))
        fig.suptitle(f"Windowed pairwise conflict set, $w={w}$ s",
                     fontsize=12.5, fontweight="bold", color=INK, y=0.985)
        fig.text(0.5, 0.015, "teal = certified inevitable collision   ·   "
                             "dashed red = dynamics-blind disc",
                 ha="center", fontsize=9, color=INK)
        fig.subplots_adjust(left=0.14, right=0.97, top=0.88, bottom=0.13)
        fig.canvas.draw()
        frames.append(Image.fromarray(
            np.asarray(fig.canvas.buffer_rgba())[..., :3]).convert(
                "P", palette=Image.ADAPTIVE, colors=96))
        plt.close(fig)

    gif = os.path.join(args.out_dir, "brt_heading.gif")
    frames[0].save(gif, save_all=True, append_images=frames[1:], loop=0,
                   duration=int(1000 / args.fps), optimize=True)
    print(f"wrote {gif} ({len(frames)} frames, "
          f"{os.path.getsize(gif) / 1e6:.1f} MB)")

    picks = [int(order[i]) for i in
             np.linspace(0, len(order) - 1, 4).round().astype(int)]
    fig, axes = plt.subplots(1, 4, figsize=(15.4, 4.2), dpi=110)
    for a, k in zip(axes, picks):
        draw(a, k, small=True)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.86, bottom=0.16, wspace=0.32)
    strip = os.path.join(args.out_dir, "brt_heading_strip.png")
    fig.savefig(strip, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {strip}")


if __name__ == "__main__":
    main()
