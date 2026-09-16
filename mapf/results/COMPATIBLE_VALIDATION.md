# Compatible-control validation — 2026-09-13

The implementation demonstrates simultaneous pairwise constraint satisfaction. The warehouse runs substantially reduce contacts relative to matched flee control, but do **not** establish better collision performance than the one-constraint ablation. Neither ablation implements Chen's MIP.

## Paired warehouse runs

Thirty seeds (0–29), 14 robots, 40 replans, three committed seconds per replan: 120 simulated seconds per episode. All arms share starts, grid tasks, disturbance streams, and the geometric planner. The HJ arms share the new 81-by-81-by-64 table, horizon 0.6 seconds, barrier margin 0.1, and world-velocity disturbance bound 0.12 per component. The control timestep is 0.1 seconds. Barrier gain is 2; rate reserve is zero.

| Controller | Mean contacts / episode | Contact-free episodes | Mean all-pair constraint violations / episode | Mean successful multi-threat agent-steps / episode |
|---|---:|---:|---:|---:|
| Geometric execution | 52.47 | 0 / 30 | Not evaluated | Not evaluated |
| Matched first-threat flee | 27.03 | 0 / 30 | Not evaluated | Not evaluated |
| One-constraint barrier | 0.73 | 19 / 30 | 1598.43 | 523.40 |
| Simultaneous barrier | 1.07 | 22 / 30 | 1438.50 | 786.37 |

Differences below are **simultaneous minus baseline** contacts, using 10,000 paired percentile-bootstrap resamples over seeds:

| Baseline | Mean paired difference | 95% bootstrap interval |
|---|---:|---:|
| Geometric | −51.40 | [−54.40, −48.47] |
| Matched flee | −25.97 | [−28.20, −23.70] |
| One-constraint barrier | +0.33 | [−0.70, +1.73] |

The interval spanning zero for the one-constraint comparison does not support a collision-improvement claim. The simultaneous arm has 32 total contacts, versus 22 for one-constraint control; 16 simultaneous-arm contacts occur in seed 28. Both controllers have state trajectories of their own, so their aggregate constraint counts are descriptive rather than an evaluation on identical encountered states.

Each episode has 16,800 agent-steps. The simultaneous filter encounters infeasibility in an average of 1438.50 of them (8.56%). Research mode continues with least-violation commands. These runs therefore violate an assumption of the conditional safety proposition and cannot validate a fleet-safety guarantee. The one-constraint arm's selected interval is infeasible in 602.53 agent-steps on average, while its returned command violates at least one all-pair constraint in 1598.43 steps.

Mean filter time is 1.47 seconds per episode for simultaneous control, versus 1.42 for one-constraint control and 4.38 for matched flee. These are implementation timings: vectorized pair evaluation and scalar membership queries have different costs.

Raw configurations, table metadata/hash, seed records, and summaries are in [compatible_comparison_30seeds.json](compatible_comparison_30seeds.json).

## Selected three-robot mechanism example

The initial state has positive barriers for all six directed pairs. Robot 0 has two active opponents. Its common feasible turn interval is [−1, −0.58765] rad/s. The one-threat filter permits −0.39305 rad/s, which violates an unselected constraint with residual −0.06084.

For three seconds at timestep 0.01 seconds, speed 1, nominal turn 1 rad/s, and zero realized disturbance (the controller still assumes bound 0.12):

| Metric | Simultaneous | One constraint |
|---|---:|---:|
| Violated agent-steps | 0 | 31 |
| Compatible multi-threat agent-steps | 38 | 4 |
| Minimum swept separation (m) | 0.98253 | 0.97300 |
| Minimum barrier | 0.26761 | 0.26003 |

Contact distance is 0.5 m. Both runs remain collision-free. The example was selected to expose a missed constraint, so it demonstrates the mechanism rather than establishing broad performance. An additional scalar example, requiring both turn ≥ 0.7 and turn ≤ −0.7, verifies explicit infeasibility handling.

See [the example data](compatible_three_robot.json) and [the figure](compatible_three_robot.svg).

## Verification and limits

The 26 automated tests cover relative dynamics against world-frame finite differences, robust affine bounds against disturbance-corner enumeration, interval feasibility and minimax fallback against independent linear programs, periodic interpolation and both sides of spatial faces, geometric enclosure, invalid caches, permutation invariance, disturbance units, common random numbers, swept contact detection, and a three-second feasible multi-threat trajectory.

A separate regression compared the old and new historical executors across 10 seeds for both unshielded and flee policies. All 20 comparisons matched continuous poses, discrete planner states, collisions, and intervention counts.

Contacts are onsets over swept **Euler segments**, not certified collision checks on continuous Dubins arcs. The new disturbance convention and contact metric differ from historical results. Physical task completion, continuous obstacle avoidance, recovery planning, and a sampled-data invariance certificate remain absent. The conditional continuous-time derivation and proposed introduction paragraph are in [COMPATIBLE_CONTROL.md](../COMPATIBLE_CONTROL.md).
