#!/usr/bin/env python
"""AMFS multi-seed statistical-validity harness (geometric vs HJ-shield).

Closes the single-seed validity gap: runs >= 30 CRN-paired seeds of the Option-A
lifelong-MAPF pipeline for both policies and reports, mirroring the WIP-forecast
harness:
  * >= 30 seeds (configurable)
  * MSER-5 warm-up truncation of the per-tick collision series (primary) +
    Welch running-mean cross-check helper
  * bootstrap 95% CIs on every headline metric
  * Common Random Numbers (CRN) across policies for PAIRED comparison
  * paired bootstrap CI for mean(hj - geom) + Holm-Bonferroni FWER control over
    the HB-1..HB-5 hypothesis family

Numpy only (loads the cached BRT). Requires cache/dubins_brt.npz.

Usage:
    python experiments/multiseed_harness.py --seeds 30 --n-agents 14
    python experiments/multiseed_harness.py --seeds 5 --total-ticks 20   # smoke
"""
from __future__ import annotations
import os
import sys
import json
import argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
for pth in (_SRC, _HERE):
    if pth not in sys.path:
        sys.path.insert(0, pth)

from exp_A_pipeline import (run_episode, make_grid, DEFAULT_BRT,
                            POLICY_GEOMETRIC, POLICY_HJ)
from executor import DubinsParams
from stats import (bootstrap_ci, paired_diff_ci, holm_bonferroni,
                   mser5, welch_running_mean)

# Headline per-seed scalar metrics for per-policy CIs.
HEADLINE = ["collisions", "coll_rate_ss", "throughput", "mean_flowtime",
            "wait_steps", "interventions"]


def steady_state_rate(coll_series):
    """MSER-5-truncated steady-state mean of the per-tick collision series."""
    s = np.asarray(coll_series, dtype=float)
    cut = mser5(s)
    tail = s[cut:] if cut < s.size else s
    return float(tail.mean()) if tail.size else float("nan"), int(cut)


def run_all(seeds, kw, params):
    """Return {policy: {metric: seed-ordered np.array}} plus raw records."""
    cols = {POLICY_GEOMETRIC: {m: [] for m in HEADLINE},
            POLICY_HJ: {m: [] for m in HEADLINE}}
    warmups = {POLICY_GEOMETRIC: [], POLICY_HJ: []}
    records = []
    for s in seeds:
        for pol in (POLICY_GEOMETRIC, POLICY_HJ):
            r = run_episode(s, pol, params=params, **kw)
            ss, cut = steady_state_rate(r["coll_series"])
            r["coll_rate_ss"] = ss
            warmups[pol].append(cut)
            for m in HEADLINE:
                cols[pol][m].append(r.get(m, float("nan")))
            records.append({k: v for k, v in r.items()
                            if k not in ("coll_series", "goal_series")})
        print(f"  seed {s} done")
    for pol in cols:
        for m in HEADLINE:
            cols[pol][m] = np.asarray(cols[pol][m], dtype=float)
    return cols, warmups, records


def aggregate(cols, rng):
    out = {}
    for pol, metrics in cols.items():
        out[pol] = {}
        for m, arr in metrics.items():
            pt, lo, hi = bootstrap_ci(arr, rng=rng)
            out[pol][m] = {"mean": pt, "ci_lo": lo, "ci_hi": hi,
                           "n": int(np.isfinite(arr).sum())}
    return out


def hypothesis_family(cols, rng):
    """HB-1..HB-5 paired diffs (hj - geom), CRN, + Holm-Bonferroni.

        HB-1 safety (total):   hj collisions   <  geom collisions
        HB-2 safety (rate):    hj coll_rate_ss <  geom coll_rate_ss (MSER-5)
        HB-3 throughput:       hj throughput  ~= geom throughput   (no loss)
        HB-4 flowtime:         hj flowtime    ~= geom flowtime      (no loss)
        HB-5 conservatism:     hj wait_steps  ~= geom wait_steps (shield != gridlock)
    """
    fam = {
        "HB-1_safety_total": "collisions",
        "HB-2_safety_rate":  "coll_rate_ss",
        "HB-3_throughput":   "throughput",
        "HB-4_flowtime":     "mean_flowtime",
        "HB-5_conservatism": "wait_steps",
    }
    results, pvals = {}, {}
    for hyp, metric in fam.items():
        a = cols[POLICY_HJ][metric]      # treatment
        b = cols[POLICY_GEOMETRIC][metric]  # baseline
        ci = paired_diff_ci(a, b, rng=rng)
        ci["metric"] = metric
        ci["contrast"] = "hj - geometric"
        results[hyp] = ci
        pvals[hyp] = ci["p_two_sided"]
    holm = holm_bonferroni(pvals)
    for hyp, h in holm.items():
        if hyp in results:
            results[hyp]["p_adj_holm"] = h["p_adj"]
            results[hyp]["reject_holm"] = h["reject"]
    return results


def print_report(agg, fam, warmups):
    print("\n" + "=" * 80)
    print("  MULTI-SEED RESULTS — per-policy bootstrap 95% CIs")
    print("=" * 80)
    for pol, metrics in agg.items():
        print(f"\n  Policy: {pol}   (MSER-5 mean warm-up cut = "
              f"{np.mean(warmups[pol]):.1f} ticks)")
        print(f"    {'metric':<16} {'mean':>12} {'95% CI':>28}  n")
        for m, v in metrics.items():
            ci = f"[{v['ci_lo']:.4g}, {v['ci_hi']:.4g}]"
            print(f"    {m:<16} {v['mean']:>12.5g} {ci:>28}  {v['n']}")
    print("\n" + "=" * 80)
    print("  HYPOTHESIS FAMILY — paired diffs hj-geom (CRN) + Holm-Bonferroni")
    print("=" * 80)
    print(f"  {'hypothesis':<20} {'metric':<15} {'diff':>10} {'95% CI':>22} "
          f"{'p_holm':>8} rej")
    for hyp, v in fam.items():
        ci = f"[{v['lo']:.3g}, {v['hi']:.3g}]"
        rej = "Y" if v.get("reject_holm") else "n"
        print(f"  {hyp:<20} {v['metric']:<15} {v['diff']:>10.4g} {ci:>22} "
              f"{v.get('p_adj_holm', float('nan')):>8.3g} {rej:>3}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--n-agents", type=int, default=14)
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--exec-steps", type=int, default=3)
    ap.add_argument("--total-ticks", type=int, default=40)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--dist-sigma", type=float, default=0.05)
    ap.add_argument("--brt", type=str, default=DEFAULT_BRT)
    ap.add_argument("--out", type=str,
                    default=os.path.join(os.path.dirname(_HERE), "results",
                                         "multiseed_results.json"))
    args = ap.parse_args()

    if not os.path.exists(args.brt):
        print(f"[FATAL] BRT cache not found: {args.brt}\n"
              f"        Run:  ./.venv/bin/python experiments/precompute_brt.py")
        sys.exit(2)

    grid = make_grid()
    params = DubinsParams(dist_sigma=args.dist_sigma)
    kw = dict(brt_path=args.brt, margin=args.margin, n_agents=args.n_agents,
              window=args.window, exec_steps=args.exec_steps,
              total_ticks=args.total_ticks, grid=grid)
    seeds = list(range(args.seed0, args.seed0 + args.seeds))
    boot_rng = np.random.default_rng(12345)

    print(f"Running {len(seeds)} CRN-paired seeds x 2 policies "
          f"(n_agents={args.n_agents}, window={args.window}, "
          f"dist_sigma={args.dist_sigma})")
    cols, warmups, records = run_all(seeds, kw, params)
    agg = aggregate(cols, boot_rng)
    fam = hypothesis_family(cols, boot_rng)
    print_report(agg, fam, warmups)

    payload = {
        "config": {"seeds": seeds, "n_agents": args.n_agents,
                   "window": args.window, "exec_steps": args.exec_steps,
                   "total_ticks": args.total_ticks, "margin": args.margin,
                   "dist_sigma": args.dist_sigma,
                   "crn": "spawn=seed, task=seed+777, dist=seed+123 (shared across policies)"},
        "aggregate_ci": agg,
        "hypothesis_family": fam,
        "warmup_cuts_mean": {k: float(np.mean(v)) for k, v in warmups.items()},
        "replications": records,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults written -> {args.out}")


if __name__ == "__main__":
    main()
