#!/usr/bin/env python
"""Generate AMFS illustration figures for the HJ-Gauss talk deck.

Produces (numpy + matplotlib only; loads the cached BRT .npz and, where needed,
re-runs a few CRN-paired episodes):
  1. warehouse_layout.png   -- RHCR-style floor: obstacles, endpoints, agents
  2. brt_slices.png         -- pairwise Dubins BRT zero-level set at theta slices
                               vs a naive collision disc (the dynamics-aware gap)
  3. collisions_bar.png     -- geometric vs HJ realized collisions, bootstrap CIs
  4. coll_series.png        -- per-tick collision series (mean over seeds) with
                               MSER-5 warm-up cut + Welch running-mean cross-check
  5. reduction_sweep.png    -- collision reduction (%) vs agent count (quick sweep)

Usage (numpy python3 is fine; needs matplotlib):
    python3 experiments/make_figures.py --outdir <slides/assets> [--seeds 12]
"""
from __future__ import annotations
import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
for p in (_SRC, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from mapf_world import WarehouseGrid, spawn_agents
from exp_A_pipeline import run_episode, make_grid, DEFAULT_BRT, POLICY_GEOMETRIC, POLICY_HJ
from executor import DubinsParams
from stats import bootstrap_ci, mser5, welch_running_mean

# Large, high-contrast plot style so figures read from the back of the room.
plt.rcParams.update({
    "font.size": 20, "font.weight": "bold",
    "axes.titlesize": 24, "axes.titleweight": "bold",
    "axes.labelsize": 22, "axes.labelweight": "bold",
    "xtick.labelsize": 16, "ytick.labelsize": 16,
    "legend.fontsize": 18, "lines.linewidth": 3.0,
    "figure.dpi": 130, "savefig.bbox": "tight",
})
GEO_C, HJ_C = "#b3402a", "#0f6f74"   # geometric vs HJ colors


def fig_warehouse(outdir, grid: WarehouseGrid):
    rng = np.random.default_rng(0)
    agents, _ = spawn_agents(grid, 14, rng)
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.imshow(~grid.obstacle, cmap="Greys", origin="lower", alpha=0.15,
              extent=[-0.5, grid.W - 0.5, -0.5, grid.H - 0.5])
    for (r, c) in np.argwhere(grid.obstacle):
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, color="#6a6fa0", alpha=0.55))
    ex = np.array(grid.endpoints)
    ax.scatter(ex[:, 1], ex[:, 0], marker="s", s=90, c="#d98a00",
               edgecolors="k", label="endpoints (pickup/dropoff)", zorder=3)
    for a in agents:
        ax.scatter(a.cell[1], a.cell[0], s=180, c=HJ_C, edgecolors="k", zorder=4)
        ax.scatter(a.goal[1], a.goal[0], marker="*", s=220, c=GEO_C,
                   edgecolors="k", zorder=4)
    ax.scatter([], [], s=180, c=HJ_C, edgecolors="k", label="agents")
    ax.scatter([], [], marker="*", s=220, c=GEO_C, edgecolors="k", label="goals")
    ax.set_title("RHCR-style structured warehouse floor (14 agents)")
    ax.set_xlabel("x (column)"); ax.set_ylabel("y (row)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    ax.set_aspect("equal")
    out = os.path.join(outdir, "warehouse_layout.png")
    fig.savefig(out); plt.close(fig); print("wrote", out)


def fig_brt_slices(outdir, brt_path):
    d = np.load(brt_path, allow_pickle=True)
    V = np.asarray(d["V"], float)
    x1, x2, th = (np.asarray(d[k], float) for k in ("x1_axis", "x2_axis", "th_axis"))
    try:
        meta = eval(str(d["meta"][0]), {"__builtins__": {}})
        cap = float(meta.get("capture_radius", 0.5))
    except Exception:
        cap = 0.5
    targets = [-np.pi / 2, 0.0, np.pi / 2]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), sharey=True)
    X1, X2 = np.meshgrid(x1, x2, indexing="ij")
    for ax, tv in zip(axes, targets):
        k = int(np.argmin(np.abs(th - tv)))
        Vk = V[:, :, k]
        cf = ax.contourf(X1, X2, Vk, levels=20, cmap="RdBu_r",
                         vmin=-np.max(np.abs(Vk)), vmax=np.max(np.abs(Vk)))
        ax.contour(X1, X2, Vk, levels=[0.0], colors="k", linewidths=3)
        ax.add_patch(Circle((0, 0), cap, fill=False, ls="--", ec="#111", lw=2.5))
        ax.set_title(rf"$\theta={tv:+.2f}$ rad")
        ax.set_xlabel(r"$x_1$ (rel.)"); ax.set_aspect("equal")
    axes[0].set_ylabel(r"$x_2$ (rel.)")
    fig.suptitle("Pairwise Dubins BRT (black = inevitable-collision boundary)  "
                 "vs naive disc (dashed)", y=1.02)
    cb = fig.colorbar(cf, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label(r"value $v^\delta$ (neg = unsafe)")
    out = os.path.join(outdir, "brt_slices.png")
    fig.savefig(out); plt.close(fig); print("wrote", out)


def _load_json(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def fig_collisions_bar(outdir, results_json):
    data = _load_json(results_json)
    if data and "aggregate_ci" in data:
        agg = data["aggregate_ci"]
        g = agg[POLICY_GEOMETRIC]["collisions"]; h = agg[POLICY_HJ]["collisions"]
        gm, glo, ghi = g["mean"], g["ci_lo"], g["ci_hi"]
        hm, hlo, hhi = h["mean"], h["ci_lo"], h["ci_hi"]
        nseeds = g.get("n", "?")
    else:
        gm = glo = ghi = hm = hlo = hhi = np.nan; nseeds = "?"
    fig, ax = plt.subplots(figsize=(9, 6.5))
    xs = [0, 1]
    ax.bar(xs, [gm, hm], color=[GEO_C, HJ_C], width=0.6,
           yerr=[[gm - glo, hm - hlo], [ghi - gm, hhi - hm]],
           capsize=10, error_kw=dict(lw=3))
    for x, m in zip(xs, [gm, hm]):
        ax.text(x, m + 2, f"{m:.1f}", ha="center", fontsize=22, fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels(["Geometric\n(dynamics-blind)", "HJ-Gauss\nshield"])
    ax.set_ylabel("Realized collisions / episode")
    red = 100 * (gm - hm) / gm if gm else float("nan")
    ax.set_title(f"HJ shield cuts realized collisions ~{red:.0f}%\n"
                 f"({nseeds} CRN-paired seeds, bootstrap 95% CI)")
    out = os.path.join(outdir, "collisions_bar.png")
    fig.savefig(out); plt.close(fig); print("wrote", out)


def fig_coll_series(outdir, brt_path, seeds, grid, kw):
    def mean_series(policy):
        arrs = []
        for s in range(seeds):
            r = run_episode(s, policy, grid=grid, **kw)
            arrs.append(np.asarray(r["coll_series"], float))
        L = min(len(a) for a in arrs)
        return np.mean([a[:L] for a in arrs], axis=0)
    gs = mean_series(POLICY_GEOMETRIC); hs = mean_series(POLICY_HJ)
    cut = mser5(gs)
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    t = np.arange(len(gs))
    ax.plot(t, gs, color=GEO_C, alpha=0.4)
    ax.plot(t, hs, color=HJ_C, alpha=0.4)
    w = 5
    ax.plot(t[w - 1:], welch_running_mean(gs, w), color=GEO_C, label="Geometric (Welch mean)")
    ax.plot(t[w - 1:], welch_running_mean(hs, w), color=HJ_C, label="HJ shield (Welch mean)")
    ax.axvline(cut, color="k", ls="--", lw=2.5, label=f"MSER-5 warm-up cut = {cut}")
    ax.set_xlabel("Rolling-horizon tick"); ax.set_ylabel("Collisions / tick")
    ax.legend(frameon=False)
    out = os.path.join(outdir, "coll_series.png")
    fig.savefig(out); plt.close(fig); print("wrote", out)


def fig_reduction_sweep(outdir, brt_path, grid, base_kw, agent_counts, seeds):
    reds = []
    for nag in agent_counts:
        kw = dict(base_kw); kw["n_agents"] = nag
        gc = hc = 0.0
        for s in range(seeds):
            gc += run_episode(s, POLICY_GEOMETRIC, grid=grid, **kw)["collisions"]
            hc += run_episode(s, POLICY_HJ, grid=grid, **kw)["collisions"]
        reds.append(100 * (gc - hc) / max(gc, 1))
    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.plot(agent_counts, reds, "-o", color=HJ_C, markersize=12)
    for x, y in zip(agent_counts, reds):
        ax.text(x, y + 1, f"{y:.0f}%", ha="center", fontsize=18, fontweight="bold")
    ax.set_xlabel("Number of agents"); ax.set_ylabel("Collision reduction (%)")
    ax.set_title(f"HJ shield collision reduction vs fleet size ({seeds} seeds/point)")
    ax.grid(alpha=0.3)
    out = os.path.join(outdir, "reduction_sweep.png")
    fig.savefig(out); plt.close(fig); print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--brt", default=DEFAULT_BRT)
    ap.add_argument("--results", default=os.path.join(
        os.path.dirname(_HERE), "results", "multiseed_results.json"))
    ap.add_argument("--seeds", type=int, default=12)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    grid = make_grid()
    params = DubinsParams(dist_sigma=0.05)
    kw = dict(brt_path=args.brt, n_agents=14, window=6, exec_steps=3,
              total_ticks=40, params=params)

    fig_warehouse(args.outdir, grid)
    fig_brt_slices(args.outdir, args.brt)
    fig_collisions_bar(args.outdir, args.results)
    fig_coll_series(args.outdir, args.brt, args.seeds, grid, kw)
    fig_reduction_sweep(args.outdir, args.brt, grid,
                        dict(brt_path=args.brt, window=6, exec_steps=3,
                             total_ticks=30, params=params),
                        agent_counts=[8, 14, 20], seeds=4)


if __name__ == "__main__":
    main()
