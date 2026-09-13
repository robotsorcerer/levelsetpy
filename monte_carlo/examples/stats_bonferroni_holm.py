#!/usr/bin/env python
"""30-seed statistical sweep + Holm-Bonferroni analysis.

Re-runs the Rockets, Dubins, and 45D multi-agent scalability examples
across 30 independent Monte Carlo seeds per condition (only the sampler's
PRNG seed varies; evaluation points and levelsetpy references are held
fixed), then runs three families of Holm-Bonferroni-corrected hypothesis
tests:

  (A) MC vs. grid discrepancy per slice (Rockets + Dubins, 6 conditions):
      paired Wilcoxon signed-rank test of the 30-seed-averaged MC field
      against the levelsetpy reference field, per condition.

  (B) L2_rel vs. Crandall-Lions bound O(sqrt(delta)) (Rockets + Dubins,
      6 conditions): one-sided one-sample test that the 30 per-seed
      L2_rel draws are significantly below sqrt(delta) = 0.28284271.

  (C) Cross-condition comparisons: Rockets-vs-Dubins at matched heading
      (3 tests), heading asymmetry theta=-pi/2 vs theta=+pi/2 within each
      system (2 tests), and pairwise speed-case comparisons for the 45D
      multi-agent case (3 tests) -- 8 tests total, corrected together.

Raw per-seed results are pickled to results/stats_raw.pkl; the summary
table and all test statistics are written to results/stats_summary.json
and printed to stdout.
"""

import sys
import os
import time
import pickle
import json
import importlib

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, "/home/lex/Documents/ML-Control-Rob/control/levelsetpy")

N_SEEDS = int(os.environ.get("STATS_N_SEEDS", 30))
BASE_SEEDS = list(range(1000, 1000 + N_SEEDS))  # arbitrary, fixed, reproducible
OUT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
#  Holm-Bonferroni step-down procedure (no statsmodels dependency)
# ═══════════════════════════════════════════════════════════════════════

def holm_bonferroni(pvals, alpha=0.05):
    """Return (adjusted_pvals, reject) aligned to the input order of pvals."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    sorted_p = pvals[order]
    adj_sorted = np.empty(m)
    running_max = 0.0
    for i in range(m):
        val = min((m - i) * sorted_p[i], 1.0)
        running_max = max(running_max, val)
        adj_sorted[i] = running_max
    adjusted = np.empty(m)
    adjusted[order] = adj_sorted
    reject = adjusted <= alpha
    return adjusted, reject


# ═══════════════════════════════════════════════════════════════════════
#  Rockets sweep
# ═══════════════════════════════════════════════════════════════════════

def sweep_rockets():
    rk = importlib.import_module("ex_rockets_3d_comparison")

    print("\n" + "=" * 70)
    print("ROCKETS: computing levelsetpy reference (once)")
    print("=" * 70)
    g_ls, v_ls, t_ls = rk.run_levelsetpy()

    xs_eval = np.linspace(*rk.SPATIAL_DOMAIN, rk.GRID_N_MC_2D)
    results = {}
    for theta_val in rk.THETA_SLICES:
        key = f"{theta_val:.4f}"
        print(f"\n--- Rockets theta={theta_val:.2f}: {N_SEEDS} seeds ---")
        V_ref = rk.interp_3d_slice(g_ls, v_ls, xs_eval, xs_eval, theta_val)
        mask = np.isfinite(V_ref)

        per_seed = []
        v_mc_stack = []
        for si, seed in enumerate(BASE_SEEDS):
            rk.MC_CFG = rk.MC_CFG._replace(seed=seed)
            t0 = time.time()
            X, Z, V_mc, history, elapsed = rk.run_mc_2d_slice(theta_val)
            V_mc = np.array(V_mc)
            diff = np.abs(V_mc - V_ref)
            l_inf = float(np.nanmax(diff[mask])) if mask.any() else float("nan")
            l2_rel = float(
                np.sqrt(np.nanmean(diff[mask] ** 2))
                / max(np.sqrt(np.nanmean(V_ref[mask] ** 2)), 1e-12)
            )
            per_seed.append({
                "seed": seed, "l_inf": l_inf, "l2_rel": l2_rel,
                "elapsed": elapsed, "iters": len(history),
                "residual": history[-1],
            })
            v_mc_stack.append(V_mc)
            print(f"    seed[{si+1}/{N_SEEDS}]={seed}: "
                  f"L_inf={l_inf:.4f} L2_rel={l2_rel:.4f} {elapsed:.1f}s")

        results[key] = {
            "theta": theta_val, "per_seed": per_seed,
            "V_ref": V_ref, "V_mc_mean": np.mean(v_mc_stack, axis=0),
            "mask": mask,
        }
    return results


# ═══════════════════════════════════════════════════════════════════════
#  Dubins sweep
# ═══════════════════════════════════════════════════════════════════════

def sweep_dubins():
    db = importlib.import_module("ex_dubins_3d_comparison")

    print("\n" + "=" * 70)
    print("DUBINS: computing levelsetpy reference (once)")
    print("=" * 70)
    g_ls, v_ls, t_ls = db.run_levelsetpy()

    xs_eval = np.linspace(*db.SPATIAL_DOMAIN, db.GRID_RES_MC)
    results = {}
    for theta_val in db.THETA_SLICES:
        key = f"{theta_val:.4f}"
        print(f"\n--- Dubins theta={theta_val:.2f}: {N_SEEDS} seeds ---")
        V_ref = db.interpolate_3d_slice(g_ls, v_ls, xs_eval, xs_eval, theta_val)
        mask = np.isfinite(V_ref)

        per_seed = []
        v_mc_stack = []
        for si, seed in enumerate(BASE_SEEDS):
            db.cfg = db.cfg._replace(seed=seed)
            X, Y, V_mc, history, elapsed = db.run_mc_solver(theta_val)
            V_mc = np.array(V_mc)
            diff = np.abs(V_mc - V_ref)
            l_inf = float(np.nanmax(diff[mask])) if mask.any() else float("nan")
            l2_rel = float(
                np.sqrt(np.nanmean(diff[mask] ** 2))
                / max(np.sqrt(np.nanmean(V_ref[mask] ** 2)), 1e-12)
            )
            per_seed.append({
                "seed": seed, "l_inf": l_inf, "l2_rel": l2_rel,
                "elapsed": elapsed, "iters": len(history),
                "residual": history[-1],
            })
            v_mc_stack.append(V_mc)
            print(f"    seed[{si+1}/{N_SEEDS}]={seed}: "
                  f"L_inf={l_inf:.4f} L2_rel={l2_rel:.4f} {elapsed:.1f}s")

        results[key] = {
            "theta": theta_val, "per_seed": per_seed,
            "V_ref": V_ref, "V_mc_mean": np.mean(v_mc_stack, axis=0),
            "mask": mask,
        }
    return results


# ═══════════════════════════════════════════════════════════════════════
#  45D multi-agent sweep (no grid ground truth: report residual/timing
#  stability across seeds, not L2_rel/L_inf).
# ═══════════════════════════════════════════════════════════════════════

def sweep_multiagent():
    ma = importlib.import_module("ex_multiagent_scalability")

    cases = {
        "A": (1.0, 2.0, "Evader faster"),
        "B": (1.0, 1.0, "Same speed"),
        "C": (2.0, 1.0, "Pursuers faster"),
    }
    results = {}
    for case_name, (a_p, a_e, label) in cases.items():
        print(f"\n--- Multiagent case {case_name} ({label}): {N_SEEDS} seeds ---")
        per_seed = []
        for si, seed in enumerate(BASE_SEEDS):
            ma.MC_CFG = ma.MC_CFG._replace(seed=seed)
            v_mc, elapsed, history = ma.run_mc_multiagent(
                f"{case_name} ({label})", a_pursuers=a_p, a_evader=a_e
            )
            v_mc = np.array(v_mc)
            per_seed.append({
                "seed": seed,
                "min_v": float(np.nanmin(v_mc)),
                "max_v": float(np.nanmax(v_mc)),
                "safe_frac": float((v_mc <= 0).sum()) / v_mc.shape[0],
                "elapsed": elapsed, "iters": len(history),
                "residual": history[-1],
            })
            print(f"    seed[{si+1}/{N_SEEDS}]={seed}: "
                  f"residual={history[-1]:.6f} {elapsed:.1f}s")
        results[case_name] = {"a_p": a_p, "a_e": a_e, "label": label,
                               "per_seed": per_seed}
    return results


# ═══════════════════════════════════════════════════════════════════════
#  Hypothesis tests
# ═══════════════════════════════════════════════════════════════════════

CRANDALL_LIONS_BOUND = float(np.sqrt(0.08))  # delta = 0.08 throughout


def test_family_A(rockets_res, dubins_res):
    """Paired Wilcoxon: 30-seed-mean MC field vs. levelsetpy reference."""
    rows, pvals = [], []
    for system, res in (("Rockets", rockets_res), ("Dubins", dubins_res)):
        for key, d in res.items():
            mask = d["mask"]
            v_mc = d["V_mc_mean"][mask]
            v_ref = d["V_ref"][mask]
            stat, p = stats.wilcoxon(v_mc.ravel(), v_ref.ravel())
            rows.append({"system": system, "theta": d["theta"],
                         "statistic": float(stat), "p_raw": float(p)})
            pvals.append(p)
    adj, reject = holm_bonferroni(pvals)
    for row, a, r in zip(rows, adj, reject):
        row["p_holm"] = float(a)
        row["reject_at_0.05"] = bool(r)
    return rows


def test_family_B(rockets_res, dubins_res):
    """One-sided one-sample test: L2_rel < Crandall-Lions bound."""
    rows, pvals = [], []
    for system, res in (("Rockets", rockets_res), ("Dubins", dubins_res)):
        for key, d in res.items():
            l2_samples = np.array([s["l2_rel"] for s in d["per_seed"]])
            stat, p_two = stats.ttest_1samp(l2_samples, CRANDALL_LIONS_BOUND)
            # one-sided (less-than): halve p if sample mean is on the
            # correct side, else it cannot be significant in that direction.
            p_one = p_two / 2 if stat < 0 else 1.0 - p_two / 2
            rows.append({
                "system": system, "theta": d["theta"],
                "mean_l2_rel": float(l2_samples.mean()),
                "std_l2_rel": float(l2_samples.std(ddof=1)),
                "bound": CRANDALL_LIONS_BOUND,
                "statistic": float(stat), "p_raw": float(p_one),
            })
            pvals.append(p_one)
    adj, reject = holm_bonferroni(pvals)
    for row, a, r in zip(rows, adj, reject):
        row["p_holm"] = float(a)
        row["reject_at_0.05"] = bool(r)
    return rows


def test_family_C(rockets_res, dubins_res, multiagent_res):
    """Cross-condition Mann-Whitney U comparisons."""
    rows, pvals = [], []

    def l2(res, key):
        return np.array([s["l2_rel"] for s in res[key]["per_seed"]])

    # (1) Rockets vs Dubins at matched heading.
    for theta_key in rockets_res.keys():
        if theta_key in dubins_res:
            a, b = l2(rockets_res, theta_key), l2(dubins_res, theta_key)
            stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            rows.append({"comparison": f"Rockets vs Dubins @ theta={rockets_res[theta_key]['theta']:.2f}",
                        "statistic": float(stat), "p_raw": float(p)})
            pvals.append(p)

    # (2) Heading asymmetry within each system: theta=-pi/2 vs theta=+pi/2.
    for system, res in (("Rockets", rockets_res), ("Dubins", dubins_res)):
        keys = sorted(res.keys(), key=lambda k: res[k]["theta"])
        neg_key, pos_key = keys[0], keys[-1]
        a, b = l2(res, neg_key), l2(res, pos_key)
        stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        rows.append({"comparison": f"{system}: theta=-pi/2 vs theta=+pi/2",
                    "statistic": float(stat), "p_raw": float(p)})
        pvals.append(p)

    # (3) Multi-agent pairwise speed-case comparisons (on final residual).
    def resid(case):
        return np.array([s["residual"] for s in multiagent_res[case]["per_seed"]])

    for c1, c2 in (("A", "B"), ("B", "C"), ("A", "C")):
        a, b = resid(c1), resid(c2)
        stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        rows.append({"comparison": f"Multiagent case {c1} vs {c2} (residual)",
                    "statistic": float(stat), "p_raw": float(p)})
        pvals.append(p)

    adj, reject = holm_bonferroni(pvals)
    for row, a, r in zip(rows, adj, reject):
        row["p_holm"] = float(a)
        row["reject_at_0.05"] = bool(r)
    return rows


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t_start = time.time()

    rockets_res = sweep_rockets()
    dubins_res = sweep_dubins()
    multiagent_res = sweep_multiagent()

    with open(os.path.join(OUT_DIR, "stats_raw.pkl"), "wb") as f:
        pickle.dump({
            "rockets": rockets_res, "dubins": dubins_res,
            "multiagent": multiagent_res, "n_seeds": N_SEEDS,
            "base_seeds": BASE_SEEDS,
        }, f)

    fam_a = test_family_A(rockets_res, dubins_res)
    fam_b = test_family_B(rockets_res, dubins_res)
    fam_c = test_family_C(rockets_res, dubins_res, multiagent_res)

    def summarize(res):
        out = {}
        for key, d in res.items():
            l2 = np.array([s["l2_rel"] for s in d["per_seed"]])
            linf = np.array([s["l_inf"] for s in d["per_seed"]])
            t = np.array([s["elapsed"] for s in d["per_seed"]])
            resid = np.array([s["residual"] for s in d["per_seed"]])
            out[key] = {
                "theta": d["theta"],
                "l2_rel_mean": float(l2.mean()), "l2_rel_std": float(l2.std(ddof=1)),
                "l_inf_mean": float(linf.mean()), "l_inf_std": float(linf.std(ddof=1)),
                "time_mean": float(t.mean()), "time_std": float(t.std(ddof=1)),
                "residual_mean": float(resid.mean()), "residual_std": float(resid.std(ddof=1)),
                "iters": d["per_seed"][0]["iters"],
            }
        return out

    def summarize_multiagent(res):
        out = {}
        for case, d in res.items():
            t = np.array([s["elapsed"] for s in d["per_seed"]])
            resid = np.array([s["residual"] for s in d["per_seed"]])
            out[case] = {
                "label": d["label"], "a_p": d["a_p"], "a_e": d["a_e"],
                "time_mean": float(t.mean()), "time_std": float(t.std(ddof=1)),
                "residual_mean": float(resid.mean()), "residual_std": float(resid.std(ddof=1)),
                "iters": d["per_seed"][0]["iters"],
            }
        return out

    summary = {
        "n_seeds": N_SEEDS,
        "rockets": summarize(rockets_res),
        "dubins": summarize(dubins_res),
        "multiagent": summarize_multiagent(multiagent_res),
        "test_family_A_mc_vs_grid": fam_a,
        "test_family_B_l2_vs_crandall_lions": fam_b,
        "test_family_C_cross_condition": fam_c,
    }

    with open(os.path.join(OUT_DIR, "stats_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("SUMMARY (mean ± std over", N_SEEDS, "seeds)")
    print("=" * 70)
    for system, res in (("Rockets", summary["rockets"]), ("Dubins", summary["dubins"])):
        for key, d in res.items():
            print(f"{system:8s} theta={d['theta']:+.2f}  "
                  f"L_inf={d['l_inf_mean']:.3f}±{d['l_inf_std']:.3f}  "
                  f"L2_rel={d['l2_rel_mean']:.3f}±{d['l2_rel_std']:.3f}  "
                  f"time={d['time_mean']:.1f}±{d['time_std']:.1f}s  "
                  f"residual={d['residual_mean']:.4f}±{d['residual_std']:.4f}  "
                  f"iters={d['iters']}")
    for case, d in summary["multiagent"].items():
        print(f"Multiagent {case} ({d['label']}): "
              f"time={d['time_mean']:.1f}±{d['time_std']:.1f}s  "
              f"residual={d['residual_mean']:.4f}±{d['residual_std']:.4f}  "
              f"iters={d['iters']}")

    print("\n--- Family A: MC vs grid discrepancy (Wilcoxon signed-rank) ---")
    for row in fam_a:
        print(f"  {row['system']} theta={row['theta']:+.2f}: "
              f"p_raw={row['p_raw']:.2e} p_holm={row['p_holm']:.2e} "
              f"reject={row['reject_at_0.05']}")

    print("\n--- Family B: L2_rel < Crandall-Lions bound (one-sample t) ---")
    for row in fam_b:
        print(f"  {row['system']} theta={row['theta']:+.2f}: "
              f"mean={row['mean_l2_rel']:.4f} bound={row['bound']:.4f} "
              f"p_raw={row['p_raw']:.2e} p_holm={row['p_holm']:.2e} "
              f"reject={row['reject_at_0.05']}")

    print("\n--- Family C: cross-condition (Mann-Whitney U) ---")
    for row in fam_c:
        print(f"  {row['comparison']}: "
              f"p_raw={row['p_raw']:.2e} p_holm={row['p_holm']:.2e} "
              f"reject={row['reject_at_0.05']}")

    print(f"\nTotal wall-clock: {(time.time() - t_start)/60:.1f} min")
    print(f"Saved raw results -> {os.path.join(OUT_DIR, 'stats_raw.pkl')}")
    print(f"Saved summary -> {os.path.join(OUT_DIR, 'stats_summary.json')}")
