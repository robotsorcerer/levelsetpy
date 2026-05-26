# Plan: Merge monte_carlo_{icml,neurips} → monte_carlo + Headline-Grade Murmuration Safety System

## Context

The `levelsetpy` repository has two Monte Carlo HJ reachability implementations that share
the same algorithm (Algorithm 1, hjgauss.pdf: quasi-linearized Cole-Hopf → Gaussian MC).
Goal: merge into `monte_carlo/`, then build a **headline-grabbing** safety certification
system for aerial starling murmurations — 4D Dubins agents, 1M+ birds on GPU, simultaneous
multi-predator attacks, all 7 IJRR23 swarm behaviors, and BRT phase-transition topology tracking.

**Empirical motivation for 1M+ aerial birds in 4D**: Real European starling murmurations
involve hundreds of thousands to millions of birds performing coordinated 3D aerial maneuvers
(swooping, swirling, cordon formation) in response to predators. Observational reference:
https://www.youtube.com/watch?v=UVko9jyAkQg&t=4s — cite this in the notes document and
`monte_carlo/README.md` as visual evidence motivating the scale and the 4D aerial model.
The video demonstrates exactly the flash-expansion, vacuole, and cordon behaviors formalized
in IJRR23.

---

## Theory Alignment Finding

Both implementations match the **value formula** (Eq. B.16) identically.
Key divergence is the **gradient formula**:

| Implementation | Formula | Paper alignment |
|---|---|---|
| `monte_carlo_neurips/src/sampling_engine.py:259-310` | `Dv = (1/(t·δ·c)) · (x − Σwᵢdᵢ/Σwᵢ)` | **Eq. B.17 ✓ exact** |
| `monte_carlo_icml/src/heat_solver.py:69-109` | `Dv = Σwᵢ·Dg(yᵢ)/Σwᵢ` | autodiff variant, not in paper |

**Decision**: merged `monte_carlo/` defaults to B.17, offers `"autodiff"` as option.

---

## New Agent Model: 4D Aerial Dubins (headline upgrade)

Classical IJRR23 agents are 3D `(x₁, x₂, θ)`. We extend to 4D aerial space:

```
State x = (x₁, x₂, x₃, θ) ∈ R² × R × S¹
  (x₁, x₂) : horizontal position (m)
  x₃        : altitude (m)
  θ          : horizontal heading angle (rad)
```

**Absolute dynamics** (extension of IJRR23 Eq. 12):
```
ẋ₁ = v·cos(θ)
ẋ₂ = v·sin(θ)
ẋ₃ = u_z              (climb/dive rate, controlled ∈ [−γ_max, γ_max])
θ̇  = ⟨w_e⟩_r         (Eq. 13 heading consensus, controlled)
```

**Relative dynamics under multi-predator attack** (extension of IJRR23 Eq. 14):
```
ẋ₁ = −v_e + v_p·cos(θ) + ⟨w_e⟩_r·x₂
ẋ₂ =  v_p·sin(θ)       − ⟨w_e⟩_r·x₁
ẋ₃ =  u_z_e − u_z_p                      (relative climb)
θ̇  =  w_p  − ⟨w_e⟩_r
```

**Capture set** (cylinder ignoring altitude, IJRR23 Eq. 38):
```
g(x) = sqrt(x₁² + x₂²) − r_capture,   r_capture = 0.2 m
```
(The altitude dimension x₃ is "ignored" in the capture cylinder — only horizontal proximity
triggers capture, as in IJRR23. This keeps the capture set exactly as in the paper while
adding the physically meaningful altitude DoF.)

**4D Murmuration Hamiltonian** (extends IJRR23 Eq. 46-47):
```
H_free(x,p) = p₁·cos(θ) + p₂·sin(θ) + p₃·u_z_e + p₄·⟨w_e⟩_r
H_att(x,p)  = p₁·(1−cos(θ)) − p₂·sin(θ)
              + p₃·(u_z_e − u_z_p)             (altitude game term)
              − ω̄_p·smooth_abs(p₄)
              + ω̄_e·smooth_abs(p₂·x₁ − p₁·x₂ + p₄)
H(x,p)      = min(H_free, H_att)               (union = min, IJRR23 Eq. 16)
```

The altitude sub-game `p₃·(u_z_e − u_z_p)` is bounded:
```
H_alt = γ_max · smooth_abs(p₃)    (evader maximizes, pursuer minimizes altitude rate)
```

---

## Multi-Predator Game (headline upgrade)

For `n_p` simultaneous predators attacking `n_f` flocks, the global Hamiltonian is
(extends IJRR23 Eq. 21 union structure):

```
H_multi(x, p) = min_{j=1}^{n_f} min_{k=1}^{n_p}  H_jk(x^{jk}, p)
```

where `x^{jk}` is the relative state of flock `j` agent under attack from predator `k`.

Safety: the **global BRT** is the union of all flock-predator BRTs:
```
Ω̄_murmur = union_{j,k} Ω̄_jk = zero sublevel set of min_{j,k} v_jk(t; x)
```

In practice, we solve the `n_f × n_p` BRT problems in parallel via JAX `vmap`.
A 1000-bird murmuration with 5 flocks × 3 predators = 15 parallel 4D BRT solves,
each for 200 birds. Full 1M-bird system: 100 flocks × 10000 birds/flock × 3 predators.

---

## 1M-Bird GPU Scaling Strategy

```
Hierarchy:
  n_f  flocks (e.g. 100)
  n_a  agents per flock (e.g. 10000)
  n_p  predators (e.g. 3)

Per solve: each (flock, predator) pair → one HJReachabilitySampler call
  - vmap over n_a × n_f × n_p evaluation points
  - pmap over available GPUs (multi-GPU)
  - Chunk size: 50k points per jit-compiled call to avoid OOM
```

**Batch solve API** in `monte_carlo/src/hjpde_solver.py`:
```python
def solve_flock_system(flocks, predators, cfg, device="gpu", chunk=50_000):
    """
    flocks: list[FlockState] — each has states: (n_a, 4)
    predators: list[PredatorState]
    Returns: safety_values: (n_f, n_a), safe_fraction, wall_time
    """
```

**`--device` argparse** in every example/test script:
```python
import argparse, jax
parser = argparse.ArgumentParser()
parser.add_argument("--device", choices=["cpu","gpu"], default="gpu")
args = parser.parse_args()
if args.device == "cpu":
    jax.config.update("jax_platform_name", "cpu")
# else default: JAX auto-selects GPU
```

---

## Phase Transition Topology Tracking (headline upgrade)

Track how the BRT zero-level-set topology changes as predator attacks evolve.
Key topological events from IJRR23: vacuole nucleation, cordon formation, flock splitting.

**Implementation** in `monte_carlo/src/topology.py`:
```python
def brt_topology_signature(v_field, grid_res=200, t_horizon=2.0):
    """
    Compute topological invariants of the BRT zero-level-set at each time step.
    Returns:
      - n_components: number of connected components of {x: v(t,x) <= 0}
      - euler_characteristic: χ = V - E + F (Euler number of the zero set)
      - betti_numbers: (β0, β1) via skimage connected_components + hole counting
    """
```

**Phase transition events** detected automatically:
| Event | Topological signature | IJRR23 behavior |
|---|---|---|
| Vacuole nucleation | χ decreases by 1 (new hole opens) | Flock splits around predator |
| Cordon formation | β₁ = 1 (BRT becomes annular) | Boundary barrier forms |
| Flash expansion | BRT radius grows monotonically | Isotropic spread away from predator |
| Flock capture | n_components increases (BRT fragments) | Agent isolated from flock |

**Visualization**: animated matplotlib figure showing BRT zero-level-set evolving through
phase transitions. Each frame = one quasi-linearization iteration. Save as `.gif` and static
publication-quality `.pdf`.

---

## Full Implementation Plan

### Step 1 — Gradient reconciliation in `heat_solver.py`

Adapt `monte_carlo_icml/src/heat_solver.py` → `monte_carlo/src/heat_solver.py`.
Add `gradient_mode: str = "b17"` to `mc_gradient_at_point` and thread through `mc_gradient_batch`.

B.17 branch (from `sampling_engine.py:259-310`):
```python
weighted_mean = jnp.sum(weights[:, None] * y, axis=0)
return (1.0 / (t_eff * delta * c)) * (x - weighted_mean)
```
Autodiff branch: existing ICML code (unchanged).

### Step 2 — Build `monte_carlo/` directory skeleton

```
monte_carlo/
├── __init__.py
├── pyproject.toml      (from icml; add murmuration, topology deps)
├── requirements.txt
├── README.md           (Step 13)
├── src/
│   ├── config.py                 ADAPT icml — add gradient_mode: str = "b17"
│   ├── transforms.py             COPY icml — unchanged
│   ├── heat_solver.py            ADAPT icml — B.17 gradient mode (Step 1)
│   ├── hjpde_solver.py           ADAPT icml — thread gradient_mode + batch flock solve
│   ├── hj_sampler.py             COPY icml — unchanged
│   ├── initial_conditions.py     COPY icml
│   ├── diagnostics.py            COPY icml
│   ├── topology.py               NEW — BRT topology tracker (Step 8)
│   └── hamiltonians/
│       ├── base.py / quadratic.py / double_integrator.py     COPY
│       ├── dubins.py / dubins_relative.py / rockets_relative.py  COPY
│       ├── murmuration.py        NEW 4D — (Step 3)
│       └── murmuration_multi.py  NEW multi-predator — (Step 4)
├── dynamics/
│   ├── dubins_jax.py / dubins_python.py   COPY neurips; strip embedded helpers
│   ├── rocket_jax.py / rocket_python.py   COPY neurips; strip embedded helpers
│   ├── config_loaders.py                  COPY neurips
│   └── murmuration_jax.py                 NEW 4D aerial + multi-predator (Step 5)
├── backends/
│   └── numpy_engine.py     COPY neurips/src/sampling_engine.py verbatim
├── examples/
│   ├── (COPY from icml/examples/)
│   └── ex_murmuration.py   NEW — 1M-bird demo + --device argparse (Step 9)
├── demos/
│   └── (COPY from neurips/demos/)
├── tests/
│   ├── (COPY + adapt icml tests)
│   └── test_murmuration_safety.py   NEW (Step 10)
└── config/
    └── rockets.cfg         COPY neurips
```

### Step 3 — 4D Murmuration Hamiltonian (`src/hamiltonians/murmuration.py`)

Template: `dubins_relative.py` (3D relative state, `smooth_abs`). Extend to 4D.

```python
class MurmuationHamiltonian4D(Hamiltonian):
    """4D aerial murmuration: state (x1, x2, x3_alt, theta). IJRR23 Eq.46-47 + altitude game."""
    def __init__(self, omega_e_bar=1., omega_p_bar=1., gamma_max=0.5,
                 n_neighbors=7, neighbor_headings=None, smoothing_eps=1e-4): ...
    def _avg_heading(self, w_e):
        # IJRR23 Eq. 13: (1/(1+n_i)) * (w_e + sum_j w_j)
        if self.neighbor_headings is None: return w_e
        return (w_e + sum(self.neighbor_headings)) / (1 + len(self.neighbor_headings))
    def __call__(self, t, x, p):
        p1,p2,p3,p4 = p[...,0], p[...,1], p[...,2], p[...,3]
        x1,x2,x3,th = x[...,0], x[...,1], x[...,2], x[...,3]
        sa = lambda z: jnp.sqrt(z**2 + self.eps**2)
        w_r = self._avg_heading(0.)
        # Altitude game: evader maximizes, pursuer minimizes altitude rate
        H_alt = self.gamma_max * sa(p3)
        H_free = p1*jnp.cos(th) + p2*jnp.sin(th) + H_alt + p4*w_r
        H_att  = (p1*(1.-jnp.cos(th)) - p2*jnp.sin(th) + H_alt
                  - self.omega_p_bar*sa(p4)
                  + self.omega_e_bar*sa(p2*x1 - p1*x2 + p4))
        return jnp.minimum(H_free, H_att)
    @property
    def state_dim(self): return 4
```

### Step 4 — Multi-predator Hamiltonian (`src/hamiltonians/murmuration_multi.py`)

```python
class MurmuationMultiPredatorHamiltonian(Hamiltonian):
    """
    n_p simultaneous predators attacking n_f flocks.
    H_multi(x, p) = min_{j,k} H_jk(x^{jk}, p)   (IJRR23 Eq. 21 union structure)
    """
    def __init__(self, n_flocks, n_predators, per_flock_hamiltonians, ...): ...
    def __call__(self, t, x, p):
        # x is (n_f * n_p, 4) stacked states; reduce via jnp.minimum
        vals = vmap(lambda H, xi: H(t, xi, p))(self.hamiltonians, x)
        return jnp.min(vals)
```

### Step 5 — 4D Aerial Dynamics (`dynamics/murmuration_jax.py`)

Functions:
- `avg_heading_jax(w_i, neighbor_headings)` — JAX Eq. 13 (mirrors `flock.py:195`)
- `abs_dynamics_4d(x, w_r, u_z, v=1.)` → `(ẋ₁, ẋ₂, ẋ₃, θ̇)` per Eq. 12 + altitude
- `rel_dynamics_4d(x, w_r_e, u_z_e, u_z_p, v_p=1., v_e=1., w_p=0.)` → Eq. 14 + altitude
- `terminal_cost_4d(x, r_capture=0.2)` → `jnp.linalg.norm(x[:2]) - r_capture` (cylinder in xy)
- `class MurmuationSolverJAX4D` — wraps `HJReachabilitySampler` with
  `MurmuationHamiltonian4D` + `terminal_cost_4d`; exposes `solve_flock_system`
- `class FlockState` — dataclass: `states: (n_a, 4)`, `flock_id: int`, `neighbor_graph`
- `class PredatorState` — dataclass: `position: (4,)`, `omega_max: float`, `gamma_max: float`

### Step 6 — `SolverConfig` additions

Add to `monte_carlo/src/config.py` (NamedTuple, defaults at end for backward compat):
```python
gradient_mode: str = "b17"          # "b17" (Eq.B.17) or "autodiff" (ICML)
chunk_size: int = 50_000            # max points per vmap call (OOM guard)
n_flocks: int = 1                   # for batch flock solve
n_predators: int = 1
```

### Step 7 — Batch flock solve in `hjpde_solver.py`

Add `solve_flock_system` method to `HJReachabilitySampler`:
```python
def solve_flock_system(self, flocks: list[FlockState],
                       predators: list[PredatorState], t: float):
    """
    vmap over all flock-predator pairs; chunk to avoid OOM.
    Returns safety_values: (n_f, n_a), safe_fraction: float, wall_time: float
    """
    # Build (n_f * n_p * n_a, 4) state array
    # Solve in chunks of cfg.chunk_size via jax.lax.map
    # Return reshaped (n_f, n_a) values
```

### Step 8 — BRT Topology Tracker (`src/topology.py`)

```python
def brt_topology_signature(v_2d_slice: jnp.ndarray):
    """
    Given a 2D slice v(x1, x2) on a grid, compute:
      - n_components: number of connected components of {v <= 0}
      - betti_1: number of holes in the BRT (β₁, using skimage label + hole detection)
      - euler_char: χ = n_components - betti_1
    """
    # Use scipy.ndimage.label on numpy(v_2d_slice <= 0)
    # Return TopologyState(n_components, betti_1, euler_char)

def detect_phase_transitions(topology_history: list[TopologyState]):
    """
    Given time series of TopologyState, detect and label:
      - 'vacuole_nucleation': χ decreases by 1
      - 'cordon_formation': betti_1 becomes 1 (annular BRT)
      - 'flock_fragmentation': n_components increases
    Returns list[(t_idx, event_name)]
    """

def animate_brt_evolution(v_slices, topology_history, save_path):
    """Matplotlib animation of 2D BRT slices with event labels. Saves .gif + .pdf."""
```

### Step 9 — Main demo script (`examples/ex_murmuration.py`)

```python
"""
1M-bird aerial murmuration safety certification demo.
Multi-predator HJI game with 4D Dubins agents.

Usage:
  python examples/ex_murmuration.py --device gpu --n-birds 1000000
  python examples/ex_murmuration.py --device cpu --n-birds 10000
"""
import argparse, time, jax, jax.numpy as jnp

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["cpu","gpu"], default="gpu")
    p.add_argument("--n-birds", type=int, default=100_000)
    p.add_argument("--n-flocks", type=int, default=10)
    p.add_argument("--n-predators", type=int, default=3)
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--n-samples", type=int, default=100_000)
    p.add_argument("--save-anim", action="store_true")
    p.add_argument("--out-dir", default="results/")
    return p.parse_args()

def main():
    args = parse_args()
    if args.device == "cpu":
        jax.config.update("jax_platform_name", "cpu")
    # ... build flocks, predators, run solve_flock_system, print timing table,
    #     optionally save BRT animation and phase-transition figure
```

**Output**: publication-quality figure with 4 panels:
1. BRT zero-level-set (xy-plane slice at mean altitude) for each flock under each predator
2. Phase-transition timeline (x=time, y=topology metric, annotated events)
3. GPU scaling curve: n_birds vs wall-clock time (log-log)
4. Safety fraction vs time for each of the 7 IJRR23 actions

### Step 10 — Safety Certification Test (`tests/test_murmuration_safety.py`)

Session-scoped fixture generates 1M states (4D) and solves BRT once:
```python
@pytest.fixture(scope="session")
def murmuration_brt_1M():
    # --device default: jax auto (GPU if available, else CPU)
    cfg = SolverConfig(delta=0.05, num_samples=100_000, max_iters=15,
                       gradient_mode="b17", chunk_size=50_000)
    key = jax.random.PRNGKey(2026)
    # 1M states in 4D: (x1,x2)∈[-5,5]², x3∈[0,100]m, θ∈[-π,π]
    states = jax.random.uniform(key, (1_000_000, 4),
        minval=jnp.array([-5.,-5.,0.,-jnp.pi]),
        maxval=jnp.array([5., 5., 100., jnp.pi]))
    H = MurmuationHamiltonian4D(n_neighbors=7)
    solver = HJReachabilitySampler(H, terminal_cost_4d, cfg)
    v = solver.solve(states, t=0.)
    return {"v": v, "states": states, "solver": solver}
```

Mark `@pytest.mark.slow`. Add `--device` argparse to `conftest.py`.

**7 action tests** (same structure as before, updated for 4D):

| Test | IJRR23 Action | Key assertion |
|---|---|---|
| `test_flock_cohesion_safety` | Formation + Cohesion | Tight-formation birds (r_xy<0.5) outside capture are safe |
| `test_heading_consensus_safety` | Global Heading Consensus | Heading-aligned birds (|θ|<0.1) at r>0.3 are safe |
| `test_predator_evasion` | Swooping/Swirling/Whirling | ≥95% of birds outside capture cylinder (r_xy>0.2) have v>0 |
| `test_flash_expansion_safety` | Flash Expansion | All birds at r_xy>1.0 have v>0 |
| `test_cordon_formation_safety` | Cordon Formation | Birds in cordon shell (1.8<r_xy<2.2) have v>0 |
| `test_vacuole_formation_safety` | Flock Splitting/Vacuole | ≥90% of birds at r_xy>0.5 have v>0 |
| `test_voronoi_inter_flock_separation` | Inter-flock Separation | ≥90% of each half-plane safe |

**Correctness tests**:
- `test_brt_finiteness` — all 1M values finite
- `test_capture_set_inside_brt` — states with r_xy<0.15 have v≤0
- `test_value_monotone_in_radius` — v increases radially from origin
- `test_phase_transitions_detected` — phase transition monitor finds ≥1 event during BRT backward solve
- `test_multi_predator_harder` — multi-predator BRT ⊇ single-predator BRT (more conservative)
- `test_altitude_decoupling` — at same (x₁,x₂,θ), higher altitude does not change safety (cylinder captures only xy)

### Step 11 — Notes document (`levelsetpy/notes/hjgauss_theory_alignment.md`)

Sections (LaTeX in `\begin{align}` blocks per CLAUDE.md, labels on every equation):
1. Viscous HJ PDE and viscosity approximation bound
2. Cole-Hopf quasi-linearization + Algorithm 1
3. Value recovery (Eq. B.16) — both implementations agree
4. Gradient estimators — divergence analysis, variance comparison (B.17 vs autodiff)
5. Frozen coefficient update and contraction theorem
6. Sample size bound (Theorem 2.5)
7. 4D aerial murmuration extension: Hamiltonian derivation, altitude sub-game
8. Multi-predator HJI game: Hamiltonian union structure
9. BRT phase transitions: topology signatures for 7 IJRR23 actions
10. Implementation decisions (merge rationale, GPU scaling strategy)

### Step 12 — `monte_carlo/README.md`

Sections: Theory (Eq. B.16/B.17/Alg1) → 4D agent model equations → Multi-predator game →
Scaling table (10k/100k/1M birds, CPU vs GPU wall-clock) → All 7 IJRR23 actions →
Deployment table → SolverConfig reference → Quick start (with `--device` flag) → References.

### Step 13 — Root `README.md` update

Append after "Citing this work":
1. **HJ-Gauss Monte Carlo Reachability** paragraph
2. Deployment table (grid-based / MC-JAX GPU / MC-NumPy debug)
3. "See [`monte_carlo/README.md`](monte_carlo/README.md) for the full HJ-Gauss deployment guide including 1M-bird murmuration safety certification."
4. Quick start bash block with `--device` flag
5. NeurIPS 2026 + ICML 2026 + IJRR23 citation stubs

---

## Execution Order

```
Step 1  — Create monte_carlo/ skeleton (__init__.py files)
Step 2  — Copy+adapt src/{config,transforms,initial_conditions,diagnostics}.py
Step 3  — Adapt src/heat_solver.py (B.17 gradient mode)
Step 4  — Adapt src/hjpde_solver.py (thread gradient_mode + solve_flock_system)
Step 5  — Copy src/hamiltonians/{base,quadratic,double_integrator,dubins,dubins_relative,rockets_relative}
Step 6  — Write src/hamiltonians/murmuration.py (4D)
Step 7  — Write src/hamiltonians/murmuration_multi.py (multi-predator)
Step 8  — Write src/topology.py (phase transition tracker)
Step 9  — Write dynamics/murmuration_jax.py (4D aerial + FlockState/PredatorState)
Step 10 — Copy+fix dynamics/{dubins_jax,dubins_python,rocket_jax,rocket_python,config_loaders}
Step 11 — Copy backends/numpy_engine.py (from sampling_engine.py verbatim)
Step 12 — Copy+adapt tests/ from icml + add murmuration tests to test_hamiltonians.py
Step 13 — Write examples/ex_murmuration.py (1M demo + --device argparse)
Step 14 — Write tests/test_murmuration_safety.py (1M, 7 actions, phase transitions)
Step 15 — Write levelsetpy/notes/hjgauss_theory_alignment.md
Step 16 — Write monte_carlo/README.md
Step 17 — Update root README.md
```

---

## Verification

```bash
# Fast tests (no 1M fixture)
pytest monte_carlo/tests/ -m "not slow" -v

# B.17 vs autodiff equivalence on quadratic cost
pytest monte_carlo/tests/test_heat_solver.py::test_b17_matches_autodiff_on_quadratic_cost -v

# 4D Hamiltonian unit tests
pytest monte_carlo/tests/test_hamiltonians.py -v -k "murmuration"

# Phase transition detection test (moderate)
pytest monte_carlo/tests/test_murmuration_safety.py::test_phase_transitions_detected -v

# Full 1M-bird certification on GPU (slow)
pytest monte_carlo/tests/test_murmuration_safety.py -m slow --device gpu -v --tb=short

# Full 1M-bird certification on CPU (very slow, reference)
pytest monte_carlo/tests/test_murmuration_safety.py -m slow --device cpu -v --tb=short

# Demo: 1M birds, 10 flocks, 3 predators, GPU, save animation
python monte_carlo/examples/ex_murmuration.py \
  --device gpu --n-birds 1000000 --n-flocks 10 --n-predators 3 --save-anim
```

---

## Critical Files

| File | Role |
|---|---|
| `monte_carlo_icml/src/heat_solver.py` | Central adaptation: add B.17 gradient mode |
| `monte_carlo_icml/src/hamiltonians/dubins_relative.py` | Template for 4D Hamiltonian |
| `monte_carlo_neurips/src/sampling_engine.py:259-310` | Authoritative B.17 gradient |
| `monte_carlo_icml/src/hjpde_solver.py` | `HJReachabilitySampler` — add `solve_flock_system` |
| `monte_carlo_icml/src/config.py` | Canonical `SolverConfig` — add `gradient_mode`, `chunk_size` |
| `levelsetpy/dynamicalsystems/flock.py:195` | Reference for `avg_heading` / Eq. 13 |

---

## Corrected Algorithm 1: Quasi-Linearization with Per-Iteration Coefficient

The following describes Algorithm 1 as implemented in `monte_carlo/src/hj_sampler.py`
after the Turn 4 fix.  The critical correction is that `c^(k)` is **recomputed at
each iteration** using the gradient and Hamiltonian from the previous iterate, and
is **clipped** (not abs-folded) to maintain sign information needed for contraction.

```
Algorithm 1 (corrected): Quasi-linear Cole-Hopf Picard iteration
-----------------------------------------------------------------
Input: eval_points x, time t, delta, c_min=1e-4, c_max=1e4
Initialize:
  v^(0) = g(x)          [terminal cost]
  c^(0) = 1/delta        [exact Cole-Hopf baseline, broadcast to (M,)]

For k = 1, 2, ..., K:
  Step 1 — Gradient recovery (Alg.1 line 3):
    Dv = mc_gradient_batch(x, t, c^(k-1))
    # Uses c^(k-1) from previous iteration, NOT frozen at 1/delta.

  Step 2 — Hamiltonian evaluation (Alg.1 line 4):
    H^(k) = H(t, x, Dv)         [per-point, shape (M,)]

  Step 3 — Coefficient update (Alg.1 line 4):
    c^(k) = (2/delta) * H^(k) / (|Dv|^2 + eps)
    c^(k) = clip(c^(k), c_min, c_max)    [CLIP, not abs]

  Step 4 — Value update:
    v^(k) = mc_value_batch(x, t, c^(k))

  Step 5 — Convergence:
    residual = ||v^(k) - v^(k-1)|| / ||v^(k-1)||
    if residual < quasi_tol: break

Output: v^(K), history=[residual_1, ..., residual_K]
```

**Key fix vs. prior code:**
- Prior: `c_grad = 1.0 / delta` (frozen); `c_frozen = abs(c_frozen) + 1e-8`
- Fixed: `c_current` updated per iteration; `c_frozen = clip(c_frozen, c_min, c_max)`

The `clip` preserves the sign of `H^(k)`, which is needed to maintain the
contraction property of the linearized operator (Assumption 2.8 of hjgauss.pdf).
Using `abs` instead silently converts negative-H regions to large positive
coefficients, breaking the contraction guarantee and increasing iteration counts.

---

## Phase Transition Markers

The following topological events are detected in `src/topology.py` and logged to
`/tmp/murmurations/phase_transitions.txt`:

| Event | Detection rule | BRT topology change |
|---|---|---|
| Vacuole nucleation | `Δχ_t < -0.5` | Euler characteristic drops; new hole opens |
| Cordon formation | `β₁(t) ≥ 1` and `β₁(t-1) = 0` | BRT becomes annular |
| Flock fragmentation | `n_comp(t) > n_comp(t-1)` | BRT splits into disjoint components |
| Flash expansion | BRT radius jump ≥ threshold OR re-unification | Isotropic outward spread |

Each event maps to one of the 7 IJRR23 swarm actions and the CDC26 phase-partition
table (CDC26 Table I).

**Output files** (per time step, written to `/tmp/murmurations/`):
```
trajectory_t{t:04d}.jpg     2D agent positions, colored by flock ID
heatmap_t{t:04d}.jpg        Value function v(x,t) as a 2D heatmap
phase_diagram_t{t:04d}.jpg  Phase space: (x1, x2) colored by heading θ
reachability_t{t:04d}.jpg   Level-set contours of v(x,t)
topology_summary.jpg        χ(t), β₁(t), n_comp(t) vs. t (updated at each step)
phase_transitions.txt       Appended event log: t, event_type, χ, β₁, n_comp
```
