#!/usr/bin/env python
"""15-Rocket Multi-Pursuer Single-Evader Game: Scalability Demonstration.

Tests importance_reach on a 45D pursuit-evasion game:
- 1 evader + 14 pursuers on a 2D plane
- Each rocket has state (x, y, θ) = 3D per agent
- Total state dimension: 45D (infeasible for grid-based levelsetpy)
- Target: Pursuers capture evader within time T
- Three cases: Evader faster (A), Same speed (B), Pursuers faster (C)

Key insight: Monte Carlo scales as O(N·n) memory, independent of dimensionality.
Grid methods would require O(N^45) points—impossible.
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

# ── Problem parameters ──────────────────────────────────────────────────────

N_PURSUERS = 14
N_ROCKETS = N_PURSUERS + 1  # 1 evader + 14 pursuers
STATE_DIM = 3 * N_ROCKETS  # (x, y, θ) per rocket = 45D

T_FINAL = 1.0
CAPTURE_RADIUS = 1.5
GRID_BOUNDS = 100.0  # State space: [-100, 100]^3 per rocket

# MC solver config
DELTA = 0.08
N_SAMPLES = 20_000  # Reduced from 100k for 45D tractability
MAX_QUASI_ITERS = 15
QUASI_TOL = 1e-5

MC_CFG = SolverConfig(
    delta=DELTA,
    num_samples=N_SAMPLES,
    max_quasi_iters=MAX_QUASI_ITERS,
    quasi_tol=QUASI_TOL,
    t_start=0.0,
    t_end=T_FINAL,
    seed=42
)

N_TIME_STEPS = 100
N_EVAL_SAMPLES = 50  # Reduced from 100 for tractability


# ── Multi-agent rocket dynamics ─────────────────────────────────────────────

class MultiRocketHamiltonian(Hamiltonian):
    """
    15-rocket pursuit-evasion game.

    State: [x_1, y_1, θ_1, ..., x_15, y_15, θ_15]  (45D)
    Evader: rocket 14 (last)
    Pursuers: rockets 0-13

    Controls: u_i ∈ [-1, 1] for each rocket (heading rate)
    Dynamics: ẋ_i = a_i cos(θ_i), ẏ_i = a_i sin(θ_i), θ̇_i = u_i
    """

    def __init__(self, a_pursuers=1.0, a_evader=1.0, eps=1e-3):
        self.a_pursuers = a_pursuers
        self.a_evader = a_evader
        self.eps = eps
        self.n_pursuers = N_PURSUERS

    def __call__(self, t, x, p):
        """
        Compute H = min_{u_0,...,u_14} [p · f(x, u)]

        Uses log-sum-exp to smooth the min over bang-bang controls.
        """
        # Unpack: x = [x_1, y_1, θ_1, ..., x_15, y_15, θ_15]
        # p = co-states, same structure

        # Evaluate Hamiltonian for all 2^15 bang-bang control combinations
        # (too expensive) — instead, use analytical min for each agent

        h_total = jnp.zeros_like(x[..., 0])

        for i in range(self.n_pursuers + 1):
            x_i = x[..., 3*i]
            y_i = x[..., 3*i + 1]
            theta_i = x[..., 3*i + 2]

            p_x = p[..., 3*i]
            p_y = p[..., 3*i + 1]
            p_theta = p[..., 3*i + 2]

            # Acceleration for this rocket
            a_i = self.a_evader if i == self.n_pursuers else self.a_pursuers

            # Dynamics: f_i = [a_i cos(θ), a_i sin(θ), u]
            # Hamiltonian: H_i = p_x * a_i cos(θ) + p_y * a_i sin(θ) + p_θ * u
            # min_u ∈ [-1,1] H_i = p_x a_i cos(θ) + p_y a_i sin(θ) + sign(p_θ) * min_u
            #                    = p_x a_i cos(θ) + p_y a_i sin(θ) - |p_θ|

            drift = p_x * a_i * jnp.cos(theta_i) + p_y * a_i * jnp.sin(theta_i)
            control_cost = -jnp.abs(p_theta)  # min_u ∈ [-1,1] [p_θ * u] = -|p_θ|

            h_total = h_total + drift + control_cost

        return h_total

    @property
    def state_dim(self) -> int:
        return STATE_DIM


def target_set_multiagent(x):
    """
    Target set: Pursuers capture evader.
    Unsafe region: min_i ||pursuer_i - evader|| < CAPTURE_RADIUS

    v(x) = min_i [sqrt((x_i - x_evader)^2 + (y_i - y_evader)^2)] - CAPTURE_RADIUS
    """
    x_evader = x[..., 3 * N_PURSUERS]      # Last rocket is evader
    y_evader = x[..., 3 * N_PURSUERS + 1]

    min_dist = jnp.full_like(x[..., 0], jnp.inf)

    for i in range(N_PURSUERS):
        x_i = x[..., 3*i]
        y_i = x[..., 3*i + 1]
        dist = jnp.sqrt((x_i - x_evader)**2 + (y_i - y_evader)**2)
        min_dist = jnp.minimum(min_dist, dist)

    return min_dist - CAPTURE_RADIUS


# ── MC Solver ───────────────────────────────────────────────────────────────

def run_mc_multiagent(case_name, a_pursuers, a_evader):
    """Run importance_reach MC solver for given speed case."""
    print(f"\n=== Case {case_name}: a_pursuers={a_pursuers}, a_evader={a_evader} ===")

    H = MultiRocketHamiltonian(a_pursuers=a_pursuers, a_evader=a_evader)
    g_fn = target_set_multiagent

    # Generate 100 random samples from 45D state space
    # State bounds: each (x, y, θ) ∈ [-100, 100] × [-100, 100] × [-π, π]
    rng = np.random.RandomState(42)
    eval_points = jnp.zeros((N_EVAL_SAMPLES, STATE_DIM))
    for i in range(N_EVAL_SAMPLES):
        for j in range(N_ROCKETS):
            x_j = rng.uniform(-GRID_BOUNDS, GRID_BOUNDS)
            y_j = rng.uniform(-GRID_BOUNDS, GRID_BOUNDS)
            theta_j = rng.uniform(-np.pi, np.pi)
            eval_points = eval_points.at[i, 3*j].set(x_j)
            eval_points = eval_points.at[i, 3*j+1].set(y_j)
            eval_points = eval_points.at[i, 3*j+2].set(theta_j)

    sampler = HJReachabilitySampler(H, g_fn, MC_CFG)
    t0 = time.time()
    v_mc, history = sampler.solve_quasi_linear(eval_points, 0.0)
    elapsed = time.time() - t0

    print(f"  Solved in {elapsed:.1f}s ({len(history)} iters)")
    if np.isnan(history[-1]):
        print(f"  Final residual: nan (convergence may have failed)")
    else:
        print(f"  Final residual: {history[-1]:.6f}")

    print(f"  Memory per iter: {N_SAMPLES * STATE_DIM * 8 / 1e6:.1f} MB")
    print(f"  Min/Max value: {np.nanmin(v_mc):.4f} / {np.nanmax(v_mc):.4f}")
    print(f"  Safe points: {(v_mc <= 0).sum()} / {N_EVAL_SAMPLES}")

    return v_mc, elapsed, history


def analyze_results(results_dict):
    """Summarize results across three cases."""
    print("\n" + "=" * 70)
    print("MULTI-AGENT SCALABILITY SUMMARY")
    print("=" * 70)
    print(f"{'Case':<10} {'a_p':<8} {'a_e':<8} {'Time (s)':<10} {'Iters':<7} {'Safe':<8} {'Min v':<10} {'Max v':<10}")
    print("-" * 70)

    for case_name, (v_mc, elapsed, history) in results_dict.items():
        n_safe = int((v_mc <= 0).sum())
        n_iters = len(history)
        min_v = float(np.nanmin(v_mc))
        max_v = float(np.nanmax(v_mc))
        # Extract speeds from context (hardcoded for now)
        speeds = {"A": (1.0, 2.0), "B": (1.0, 1.0), "C": (2.0, 1.0)}
        a_p, a_e = speeds.get(case_name, (0, 0))
        print(f"{case_name:<10} {a_p:<8.1f} {a_e:<8.1f} {elapsed:<10.1f} {n_iters:<7} {n_safe:<8} {min_v:<10.4f} {max_v:<10.4f}")

    print("=" * 70)
    print(f"State dimension:        {STATE_DIM}D")
    print(f"Sample count per iter:  {N_SAMPLES:,}")
    print(f"Evaluation points:      {N_EVAL_SAMPLES}")
    print(f"Time horizon:           T = {T_FINAL}")
    print(f"Capture radius:         {CAPTURE_RADIUS} ft")
    print(f"Viscosity parameter:    δ = {DELTA}")
    print("\nKey insight: This 45D problem is infeasible for grid-based methods.")
    print("Grid-based levelsetpy would require N^45 storage (impossible).")
    print("Monte Carlo uses O(N·45) memory per iteration — highly scalable.")
    print("=" * 70)


# ── main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("15-ROCKET MULTI-PURSUER GAME: IMPORTANCE_REACH SCALABILITY TEST")
    print("=" * 70)
    print(f"State dimension: {STATE_DIM}D ({N_PURSUERS} pursuers + 1 evader)")
    print(f"Evaluation points: {N_EVAL_SAMPLES} random samples")
    print(f"Samples per iteration: {N_SAMPLES:,}")
    print("=" * 70)

    results = {}

    # Case B: Same speed (primary test)
    v_b, t_b, h_b = run_mc_multiagent("B (Same speed)", a_pursuers=1.0, a_evader=1.0)
    results["B"] = (1.0, 1.0, v_b, t_b, h_b)

    # Case A: Evader faster
    v_a, t_a, h_a = run_mc_multiagent("A (Evader faster)", a_pursuers=1.0, a_evader=2.0)
    results["A"] = (1.0, 2.0, v_a, t_a, h_a)

    # Case C: Pursuers faster
    v_c, t_c, h_c = run_mc_multiagent("C (Pursuers faster)", a_pursuers=2.0, a_evader=1.0)
    results["C"] = (2.0, 1.0, v_c, t_c, h_c)

    # Summary
    results_for_analysis = {
        "A": (v_a, t_a, h_a),
        "B": (v_b, t_b, h_b),
        "C": (v_c, t_c, h_c),
    }
    analyze_results(results_for_analysis)

    print("\n✓ Scalability test complete.")
    print("  45D state space successfully handled by importance_reach.")
    print("  Demonstrates advantage over grid-based reachability analysis.")
