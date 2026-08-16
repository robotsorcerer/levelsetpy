# AMFS — HJ-Gauss for Multi-Agent Path Finding (code)

Experiment code for folding HJ-Gauss into MAPF (innovations #1 windowed
reachability = RHCR window, #2 dynamics-aware conflict predicate as a runtime
BRT safety shield). Design rationale + literature: `HJ_Gauss/AMFS/DESIGN.md`.

## Layout
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

## Reproduce
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

## Headline result (30 seeds, n=14, dist_sigma=0.05)
Realized collisions: geometric 97.7 [90.8, 104.9] vs HJ-shield 66.1 [61.2, 71.2].
Paired hj-geom = -31.6 [-38.0, -25.4], Holm p≈0 (REJECT) — ~32% fewer collisions.
Throughput / flowtime unchanged (see DESIGN limitations: discrete planner is
authoritative for task progress, so the shield's cost is not yet modeled).
