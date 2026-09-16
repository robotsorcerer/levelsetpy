#!/usr/bin/env python3
"""Grid (level-set) solve of the pairwise Dubins BRT -> BRTPredicate .npz cache.

Companion to `precompute_brt.py`, which solves the same tube with the HJ-Gauss
Monte Carlo sampler. The pairwise relative state is 3D, where a structured grid
solve is both cheaper and the accuracy reference (cf. LevelSetPy); the sampler is
retained as an ablation and for the >pairwise extension.

Writes the same keys as `hj_conflict.precompute_brt` (V, x1_axis, x2_axis,
th_axis, meta) so `BRTPredicate` consumes either cache unchanged.

Dynamics/Hamiltonian are `levelsetpy.dynamicalsystems.DubinsVehicleRel`, whose
convention matches `monte_carlo/src/hamiltonians/dubins_relative.py` exactly:
    H = p1[v_e - v_p cos x3] - p2[v_p sin x3] - w|p1 x2 - p2 x1 - p3| + w|p3|

Usage
-----
    python experiments/precompute_brt_grid.py --horizon 0.6 --capture-radius 0.5 \
        --res 61 --n-theta 41 --out cache/dubins_brt_grid.npz
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from math import pi
from os.path import abspath, join

import numpy as np
import torch

# levelsetpy lives two directories up from monte_carlo/
_LSPY_ROOT = abspath(join(os.path.dirname(__file__), "..", "..", ".."))
if _LSPY_ROOT not in sys.path:
    sys.path.insert(0, _LSPY_ROOT)

from levelsetpy.grids import createGrid                                # noqa: E402
from levelsetpy.dynamicalsystems import DubinsVehicleRel               # noqa: E402
from levelsetpy.initialconditions import shapeCylinder                 # noqa: E402
from levelsetpy.spatialderivative import upwindFirstENO2               # noqa: E402
from levelsetpy.explicitintegration.integration import odeCFL3, odeCFLset  # noqa: E402
from levelsetpy.explicitintegration.dissipation import artificialDissipationGLF  # noqa: E402
from levelsetpy.explicitintegration.term import (                      # noqa: E402
    termRestrictUpdate, termLaxFriedrichs)
from levelsetpy.utilities import Bundle, expand, eps                   # noqa: E402


def solve(horizon=0.6, capture_radius=0.5, x_lim=4.0, res=61, n_theta=41,
          speed=1.0, turn_rate=1.0, n_substeps=10, verbose=True):
    """Solve the windowed pairwise BRT on a grid. Returns (V, axes, seconds)."""
    grid_min = expand(np.array((-x_lim, -x_lim, -pi)), ax=1)
    grid_max = expand(np.array((x_lim, x_lim, pi)), ax=1)
    pdDims = 2                                     # heading is periodic
    N = np.array([[res, res, n_theta]]).T.astype(int)
    grid_max[2, 0] *= (1 - 2 / N[2, 0])            # avoid duplicating -pi/+pi
    g = createGrid(grid_min, grid_max, N, pdDims)

    # target: capture cylinder of radius r_c in (x1,x2), free in heading
    value_init = shapeCylinder(g, 2, np.zeros((3, 1)), capture_radius)

    dyn = DubinsVehicleRel(g, speed, turn_rate)
    g.xs = [torch.as_tensor(x) for x in g.xs]

    fd = Bundle(dict(
        innerFunc=termLaxFriedrichs,
        innerData=Bundle({'grid': g, 'hamFunc': dyn.hamiltonian,
                          'partialFunc': dyn.dissipation,
                          'dissFunc': artificialDissipationGLF,
                          'CoStateCalc': upwindFirstENO2}),
        positive=False,        # grow the set inward in backward time => a TUBE
    ))

    options = Bundle(dict(factorCFL=0.95, stats='off', singleStep='off'))
    t_range = [0.0, float(horizon)]
    t_steps = (t_range[1] - t_range[0]) / n_substeps
    small = 100 * eps

    value = torch.as_tensor(copy.copy(value_init))
    t_now = t_range[0]
    t0 = time.perf_counter()
    while (t_range[1] - t_now) > small * max(t_range[1], 1.0):
        t_span = np.hstack([t_now, min(t_range[1], t_now + t_steps)])
        t, y, _ = odeCFL3(termRestrictUpdate, t_span, value.flatten(),
                          odeCFLset(options), fd)
        t_now = t
        value = y.reshape(g.shape)
        if verbose:
            print(f"  t={t_now:.3f}/{t_range[1]}  |V|_2={float(torch.linalg.norm(y)):.3f}",
                  flush=True)
    secs = time.perf_counter() - t0

    V = value.cpu().numpy().astype(np.float32)
    axes = [np.asarray(g.vs[i]).ravel().astype(float) for i in range(3)]
    return V, axes, secs, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=float, default=0.6)
    ap.add_argument("--capture-radius", type=float, default=0.5)
    ap.add_argument("--x-lim", type=float, default=4.0)
    ap.add_argument("--res", type=int, default=61, help="points per (x1,x2) axis")
    ap.add_argument("--n-theta", type=int, default=41)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--turn-rate", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = args.out or abspath(join(os.path.dirname(__file__), "..", "cache",
                                  f"dubins_brt_grid_{args.res}x{args.n_theta}.npz"))

    print(f"[grid] solving {args.res}x{args.res}x{args.n_theta} "
          f"on [-{args.x_lim},{args.x_lim}]^2 x [-pi,pi), horizon={args.horizon}, "
          f"r_c={args.capture_radius}")
    V, axes, secs, g = solve(horizon=args.horizon,
                             capture_radius=args.capture_radius,
                             x_lim=args.x_lim, res=args.res,
                             n_theta=args.n_theta, speed=args.speed,
                             turn_rate=args.turn_rate)

    frac = float((V <= 0).mean())
    print(f"[grid] done in {secs:.1f}s   frac_inside={frac:.4f}")

    meta = dict(solver="levelsetpy-grid", window_horizon=args.horizon,
                capture_radius=args.capture_radius, x_lim=args.x_lim,
                n_xy=args.res, n_theta=args.n_theta,
                v_p=args.speed, v_e=args.speed, w=args.turn_rate,
                scheme="ENO2/GLF/odeCFL3", factorCFL=0.95,
                solve_seconds=secs, frac_inside=frac)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, V=V, x1_axis=axes[0], x2_axis=axes[1],
                        th_axis=axes[2],
                        meta=np.array([repr(meta)], dtype=object))
    print(f"[grid] cached -> {out}")


if __name__ == "__main__":
    main()
