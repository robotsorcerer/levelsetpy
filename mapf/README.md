# HJ-MAPF — HJ-Gauss for Multi-Agent Path Finding (code)

## Simultaneous pairwise control

The new filter intersects the steering constraints for all nearby opponents
and projects the nominal turn rate into the common feasible interval. It reports
incompatible constraints and uses a new table with the executor's ego-body
coordinates. See [COMPATIBLE_CONTROL.md](COMPATIBLE_CONTROL.md) for the derivation,
conditional safety statement, failure handling, and limitations.

```bash
PYTHONPATH=$PWD python3 experiments/precompute_brt_grid.py \
  --horizon 0.6 --capture-radius 0.5 --x-lim 4.0 \
  --res 81 --n-theta 64 --speed 1.0 --turn-rate 1.0 \
  --out cache/dubins_brt_81x64.npz
pytest -q
python3 experiments/compatible_control_demo.py --plot
python3 experiments/compare_shields.py --seeds 30 --n-agents 14 --total-ticks 40
python3 experiments/make_brt_animation.py \
  --brt cache/dubins_brt_81x64.npz --out-dir /path/to/paper/figures
cd /path/to/ICRA2027 && pdflatex -interaction=nonstopmode icra27.tex
```

The commands above reproduce the revised BRT cache, validation runs, and the
2x2 Figure 3 JPG used by the paper. Set `/path/to/paper/figures` to the
paper's `figures/` directory.

The comparison uses matched velocity disturbances and reports infeasible
agent-steps as well as collisions. The one-constraint ablation is not Chen's MIP.
Sample-time feasibility is not a fleet-safety certificate. Continuous obstacle
avoidance, recovery planning, and physical task completion remain unimplemented.

## Historical layout
```
src/hj_conflict.py     BRT precompute (JAX) + numpy windowed-BRT membership predicate
src/mapf_world.py      RHCR-style warehouse grid + lifelong task assignment
src/planner.py         windowed prioritized planning (geometric + optional HJ predicate)
src/executor.py        continuous Dubins rollout + bounded disturbance + BRT flee-shield
src/stats.py           pure-numpy bootstrap CI / paired-diff / Holm-Bonferroni / MSER-5 / Welch
experiments/precompute_brt.py    one-time BRT cache (needs JAX venv)
experiments/exp_A_pipeline.py    single-seed paired trial (geometric vs hj)
experiments/multiseed_harness.py >=30-seed CRN-paired validation
cache/dubins_brt.npz   cached pairwise Dubins BRT (created by precompute)
results/               harness JSON output
infra/lambda_labs_notes.md   GPU scaling onboarding for Option C
```

## Reproduce historical experiments
```bash
# 0. one-time JAX venv (CPU)
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python jax scipy numpy matplotlib

# 1. precompute the BRT (needs JAX) — ~10s
./.venv/bin/python experiments/precompute_brt.py --capture-radius 0.5

# 2. single-seed sanity trial (numpy only)
python3 experiments/exp_A_pipeline.py --seed 0 --n-agents 14

# 3. full statistical validation (numpy only) — ~7 min CPU
python3 experiments/multiseed_harness.py --seeds 30 --n-agents 14 --total-ticks 40
```

## Historical result (30 seeds, n=14, dist_sigma=0.05)
Realized collisions: geometric 97.7 [90.8, 104.9] vs HJ-shield 66.1 [61.2, 71.2].
Paired hj-geom = -31.6 [-38.0, -25.4], Holm p≈0 (REJECT) — ~32% fewer collisions.
Throughput / flowtime unchanged (see DESIGN limitations: discrete planner is
authoritative for task progress, so the shield's cost is not yet modeled).
