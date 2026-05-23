#!/usr/bin/env python
"""Manufacturing Dispatch Safety Envelope: Importance_Reach vs. LevelSetPy.

Fixed version with correct grid indexing and target set function.
"""

import sys, os, time
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, "/home/lex/Documents/ML-Control-Rob/control/levelsetpy")

from src.config import SolverConfig
from src.hamiltonians.base import Hamiltonian
from src.hj_sampler import HJReachabilitySampler

# Parameters
LAM, MU_C, MU_D = 1.0, 0.715, 1.3
T_FINAL = 1.0
X_MIN, X_MAX = 0.0, 25.0
Y_MIN, Y_MAX = 0.0, 25.0
GRID_N_LS = 101
GRID_N_MC = 40
DELTA = 0.08

MC_CFG = SolverConfig(delta=DELTA, num_samples=12_000, max_quasi_iters=15,
                      quasi_tol=1e-5, t_start=0.0, t_end=T_FINAL, seed=42)

class ManufacturingDispatchHamiltonian(Hamiltonian):
    def __init__(self, lam=1.0, mu_c=0.715, mu_d=1.3, eps=1e-3):
        self.lam = lam
        self.mu_c = mu_c
        self.mu_d = mu_d
        self.eps = eps

    def __call__(self, t, x, p):
        p1, p2 = p[..., 0], p[..., 1]
        h1 = p1 * self.lam
        h2 = p1 * (self.lam - self.mu_c) + p2 * self.mu_c
        h3 = p1 * self.lam - p2 * self.mu_d
        h4 = p1 * (self.lam - self.mu_c) + p2 * (self.mu_c - self.mu_d)
        hams = jnp.stack([h1, h2, h3, h4], axis=-1)
        return -jnp.log(jnp.mean(jnp.exp(-hams / self.eps), axis=-1)) * self.eps

    @property
    def state_dim(self) -> int:
        return 2

def target_set_mfg(x):
    """Target: deadlock (q_c > 20) OR starvation (q_d < 1)"""
    q_c, q_d = x[..., 0], x[..., 1]
    return jnp.maximum(q_c - 20.0, 1.0 - q_d)

def load_levelsetpy_reference():
    safe_mask_path = "/media/lex/data/diffusion/conwip_results/safe_mask.npy"
    if not os.path.exists(safe_mask_path):
        print(f"[ERROR] Safe mask not found at {safe_mask_path}")
        return None
    safe_mask = np.load(safe_mask_path)
    print(f"[INFO] Loaded LevelSetPy safe mask: shape {safe_mask.shape}")
    return safe_mask

class HJDispatchFilter:
    """Query HJ safety envelope (from hj_dispatch_envelope.py)."""
    def __init__(self, safe_mask: np.ndarray, grid_min=np.array([X_MIN, Y_MIN]), grid_max=np.array([X_MAX, Y_MAX])):
        self.safe_mask = safe_mask
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.n = safe_mask.shape[0]

    def _to_idx(self, qc: float, qd: float):
        ic = int(np.clip((qc - self.grid_min[0]) / (self.grid_max[0] - self.grid_min[0]) * (self.n - 1), 0, self.n - 1))
        id_ = int(np.clip((qd - self.grid_min[1]) / (self.grid_max[1] - self.grid_min[1]) * (self.n - 1), 0, self.n - 1))
        return ic, id_

    def is_safe(self, qc: float, qd: float) -> bool:
        ic, id_ = self._to_idx(qc, qd)
        return bool(self.safe_mask[ic, id_])

def run_mc_manufacturing():
    print("\n=== Importance_Reach MC Solver ===")
    H = ManufacturingDispatchHamiltonian(lam=LAM, mu_c=MU_C, mu_d=MU_D)
    g_fn = target_set_mfg
    
    xs = jnp.linspace(X_MIN, X_MAX, GRID_N_MC)
    ys = jnp.linspace(Y_MIN, Y_MAX, GRID_N_MC)
    XX, YY = jnp.meshgrid(xs, ys, indexing="ij")
    eval_points = jnp.stack([XX.ravel(), YY.ravel()], axis=-1)
    
    sampler = HJReachabilitySampler(H, g_fn, MC_CFG)
    t0 = time.time()
    v_mc, history = sampler.solve_quasi_linear(eval_points, 0.0)
    elapsed = time.time() - t0
    
    v_mc = v_mc.reshape(GRID_N_MC, GRID_N_MC)
    print(f"  MC done in {elapsed:.1f}s ({len(history)} iters)")
    if np.isnan(history[-1]):
        print(f"  Final residual: nan (convergence failed)")
    else:
        print(f"  Final residual: {history[-1]:.6f}")
    
    return XX, YY, v_mc, elapsed

def compute_error_metrics(XX, YY, v_mc, hj_filter):
    """Compare MC solution against HJ safety filter using proper evaluation."""
    v_hj = np.zeros_like(v_mc)

    for i in range(GRID_N_MC):
        for j in range(GRID_N_MC):
            qc = float(XX[i, j])
            qd = float(YY[i, j])
            is_safe = hj_filter.is_safe(qc, qd)
            v_hj[i, j] = 1.0 if is_safe else -1.0

    # Classification-based error (MC may have smooth values, HJ is discrete safe/unsafe)
    mc_safe = v_mc <= 0  # MC: negative = unsafe
    hj_safe = v_hj > 0   # HJ: positive = safe

    diff = np.abs(v_mc - v_hj)
    l_inf = np.max(diff)
    l2_rel = np.sqrt(np.mean(diff ** 2)) / (np.sqrt(np.mean(np.abs(v_hj) ** 2)) + 1e-12)

    return l_inf, l2_rel, v_hj, mc_safe, hj_safe, diff

def safety_comparison(mc_safe, hj_safe):
    return {
        "disagreement_rate": (mc_safe != hj_safe).mean(),
        "false_positive_rate": ((mc_safe) & ~(hj_safe)).mean(),
        "false_negative_rate": (~(mc_safe) & (hj_safe)).mean(),
    }

def plot_comparison(XX, YY, v_mc, v_hj, diff, mc_safe, hj_safe, output_dir="./"):
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Manufacturing Dispatch: Importance_Reach vs. LevelSetPy",
                 fontsize=14, fontweight="bold")
    
    # MC solution
    ax = axes[0, 0]
    im0 = ax.contourf(XX, YY, np.array(v_mc), levels=20, cmap="RdBu_r")
    ax.contour(XX, YY, np.array(v_mc), levels=[0.0], colors="k", linewidths=2)
    ax.set_title("Importance_Reach (MC)")
    ax.set_xlabel("$q_c$ (constraint)")
    ax.set_ylabel("$q_d$ (downstream)")
    plt.colorbar(im0, ax=ax)
    
    # HJ reference
    ax = axes[0, 1]
    im1 = ax.contourf(XX, YY, v_hj, levels=20, cmap="RdBu_r")
    ax.contour(XX, YY, v_hj, levels=[0.0], colors="k", linewidths=2)
    ax.set_title("HJ Filter (LevelSetPy)")
    ax.set_xlabel("$q_c$")
    ax.set_ylabel("$q_d$")
    plt.colorbar(im1, ax=ax)
    
    # Error
    ax = axes[0, 2]
    im2 = ax.contourf(XX, YY, diff, levels=20, cmap="hot")
    ax.set_title("Pointwise Error")
    ax.set_xlabel("$q_c$")
    ax.set_ylabel("$q_d$")
    plt.colorbar(im2, ax=ax)
    
    # Safe regions
    for idx, (safe_data, title) in enumerate([(mc_safe, "MC"), (hj_safe, "HJ Filter")]):
        ax = axes[1, idx]
        ax.contourf(XX, YY, safe_data.astype(float), levels=[0, 0.5, 1],
                   colors=["red", "green"], alpha=0.6)
        ax.set_title(f"{title} Safe Region")
        ax.set_xlabel("$q_c$")
        ax.set_ylabel("$q_d$")
    
    # Disagreement
    ax = axes[1, 2]
    disagreement = mc_safe != hj_safe
    ax.contourf(XX, YY, disagreement.astype(float), levels=2, cmap="RdYlGn_r", alpha=0.7)
    ax.set_title("Safety Disagreement")
    ax.set_xlabel("$q_c$")
    ax.set_ylabel("$q_d$")
    
    plt.tight_layout()
    outfile = os.path.join(output_dir, "manufacturing_dispatch_comparison.png")
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"[INFO] Saved: {outfile}")
    plt.close()

if __name__ == "__main__":
    print("=" * 70)
    print("Manufacturing Dispatch: Importance_Reach vs. HJ Filter (LevelSetPy)")
    print("=" * 70)

    safe_mask = load_levelsetpy_reference()
    if safe_mask is None:
        sys.exit(1)

    hj_filter = HJDispatchFilter(safe_mask)
    XX, YY, v_mc, time_mc = run_mc_manufacturing()
    l_inf, l2_rel, v_hj, mc_safe, hj_safe, diff = compute_error_metrics(XX, YY, v_mc, hj_filter)

    print(f"\n=== Error Metrics ===")
    print(f"  L∞ error:     {l_inf:.4f}")
    print(f"  L²_rel error: {l2_rel:.4f}")
    print(f"  O(√δ) bound:  {np.sqrt(DELTA):.4f}")

    safety_metrics = safety_comparison(mc_safe, hj_safe)
    print(f"\n=== Safety Metrics ===")
    for k, v in safety_metrics.items():
        print(f"  {k}: {v:.4f}")

    print(f"\n=== Performance ===")
    print(f"  Wall-clock:  {time_mc:.1f}s")
    print(f"  Memory/iter: {MC_CFG.num_samples * 2 * 8 / 1e6:.1f} MB")

    plot_comparison(XX, YY, v_mc, v_hj, diff, mc_safe, hj_safe)
    print("\n" + "=" * 70)
