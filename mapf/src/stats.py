"""Pure-numpy statistical-validity helpers for the AMFS harness.

Mirrors the API of the user's sim_wip_forecast/eval/multiseed_harness.py so the
two studies report statistics identically:
  * bootstrap_ci        -- percentile bootstrap (1-alpha) CI
  * paired_diff_ci      -- paired bootstrap CI for mean(a-b) under CRN
  * holm_bonferroni     -- step-down FWER control over a hypothesis family
  * mser5               -- MSER-5 warm-up truncation (White 1997)
  * welch_running_mean  -- Welch graphical cross-check (moving average)

Dependency-light by design: numpy only (no scipy).
"""
from __future__ import annotations
import numpy as np


def bootstrap_ci(x, n_boot: int = 10_000, alpha: float = 0.05,
                 statistic=np.mean, rng=None):
    """Percentile bootstrap (1-alpha) CI for `statistic`. Returns (pt, lo, hi)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return (float("nan"),) * 3
    if n == 1:
        v = float(statistic(x))
        return (v, v, v)
    if rng is None:
        rng = np.random.default_rng(0)
    point = float(statistic(x))
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = statistic(x[idx], axis=1)
    return (point, float(np.quantile(boot, alpha / 2)),
            float(np.quantile(boot, 1 - alpha / 2)))


def paired_diff_ci(a, b, n_boot: int = 10_000, alpha: float = 0.05, rng=None):
    """Paired bootstrap CI for mean(a-b) where a[i], b[i] share seed i (CRN)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape, "paired_diff_ci needs equal-length CRN-paired arrays"
    d = a - b
    d = d[np.isfinite(d)]
    n = d.size
    if n == 0:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "p_two_sided": float("nan"), "n": 0}
    if rng is None:
        rng = np.random.default_rng(0)
    point = float(d.mean())
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = d[idx].mean(axis=1)
    lo = float(np.quantile(boot, alpha / 2))
    hi = float(np.quantile(boot, 1 - alpha / 2))
    p = 2.0 * min((boot <= 0).mean(), (boot >= 0).mean())
    return {"diff": point, "lo": lo, "hi": hi,
            "p_two_sided": float(min(1.0, p)), "n": int(n)}


def holm_bonferroni(pvalues: dict, alpha: float = 0.05) -> dict:
    """Holm-Bonferroni step-down FWER control over {name: p}."""
    items = [(k, float(v)) for k, v in pvalues.items() if np.isfinite(v)]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    out, prev_adj, still = {}, 0.0, True
    for i, (name, p) in enumerate(items):
        adj = max(min(1.0, (m - i) * p), prev_adj)
        prev_adj = adj
        reject = still and (adj <= alpha)
        if not reject:
            still = False
        out[name] = {"p_raw": p, "p_adj": adj, "reject": bool(reject), "rank": i + 1}
    for k, v in pvalues.items():
        if k not in out:
            out[k] = {"p_raw": float(v), "p_adj": float("nan"),
                      "reject": False, "rank": -1}
    return out


def mser5(series: np.ndarray) -> int:
    """MSER-5 warm-up truncation (White 1997). Returns cut index in series units."""
    s = np.asarray(series, dtype=float)
    s = s[np.isfinite(s)]
    n = s.size
    if n < 10:
        return 0
    b = 5
    nb = n // b
    batched = s[:nb * b].reshape(nb, b).mean(axis=1)
    best_d, best_val = 0, np.inf
    for d in range(0, nb - 5):
        tail = batched[d:]
        val = tail.var(ddof=1) / tail.size
        if val < best_val:
            best_val, best_d = val, d
    return int(best_d * b)


def welch_running_mean(series: np.ndarray, window: int = 10) -> np.ndarray:
    """Welch graphical cross-check: moving average of the series."""
    s = np.asarray(series, dtype=float)
    if s.size < window:
        return s.copy()
    return np.convolve(s, np.ones(window) / window, mode="valid")
