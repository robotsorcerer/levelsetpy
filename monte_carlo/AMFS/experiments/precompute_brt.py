#!/usr/bin/env python
"""One-time precompute of the pairwise Dubins pursuit-evasion BRT (AMFS).

Run under the AMFS JAX venv:
    ./.venv/bin/python experiments/precompute_brt.py

Caches cache/dubins_brt.npz used by the (numpy-only) MAPF pipeline + harness.
The value is solved at t = window_horizon (innovation #1: reachability window
== RHCR rolling window). capture_radius / speeds / turn-rate match the executor
DubinsParams so the certificate and the ground-truth dynamics are consistent.
"""
import os
import sys
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from hj_conflict import precompute_brt

DEFAULT_OUT = os.path.join(os.path.dirname(_HERE), "cache", "dubins_brt.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=DEFAULT_OUT)
    ap.add_argument("--window-horizon", type=float, default=0.6)
    ap.add_argument("--capture-radius", type=float, default=0.8)
    ap.add_argument("--num-samples", type=int, default=3000)
    ap.add_argument("--n-xy", type=int, default=31)
    ap.add_argument("--n-theta", type=int, default=21)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--max-quasi-iters", type=int, default=10)
    args = ap.parse_args()

    precompute_brt(
        args.out,
        window_horizon=args.window_horizon,
        capture_radius=args.capture_radius,
        num_samples=args.num_samples,
        n_xy=args.n_xy,
        n_theta=args.n_theta,
        delta=args.delta,
        max_quasi_iters=args.max_quasi_iters,
        v_p=-1.0, v_e=1.0, w=1.0,
    )


if __name__ == "__main__":
    main()
