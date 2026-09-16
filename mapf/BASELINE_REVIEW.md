# External baseline protocol and verification

Sources:
- Chen, Shih, Tomlin, CDC 2016, sections III–V and equations (4)–(7):
  https://arxiv.org/html/1603.05200
- CCBS authors' implementation and MIT license:
  https://github.com/PathPlanning/Continuous-CBS/tree/b8f45e166cf39ee7e8e06f0bfe4fbab0dee68932
- Walker, Sturtevant, Felner, IJCAI 2018, *Extended Increasing Cost Tree Search
  for Non-Unit Cost Domains* (`pdfs/MAPF_R.pdf` in the paper checkout).

## Two experiment groups

1. Lifelong controller comparison: seeds 0–29, identical committed geometric
   plans and disturbance draws, 14 agents, 40 replans × 3 seconds. The existing
   four arms are rerun and checked against saved integer/contact diagnostics.
   Two Chen-assignment thresholds are added. A separate 3-agent comparison uses
   the priority matrix published by Chen; no larger-fleet priority extension
   enters that case.
2. CCBS MAPF_R snapshots: seeds 0–29, 14 agents, same warehouse, unique goals,
   cardinal graph edges, radius 0.25, continuous wait times, 10 s solver cap.
   The planner is unmodified upstream CCBS with PC, DS, and greedy high-level
   heuristic enabled. We report all failures. Three execution arms share each
   solved schedule, horizon, initial headings (zero), and noise draws. The
   nominal schedule is checked separately from the disturbed unicycle rollout.
   Runtime controllers can trade tracking accuracy for fewer pair contacts;
   final mean goal error is reported for that reason. Physical obstacle safety
   is not assessed here.

CCBS is not passed through a discrete-time path rounding adapter. The rollout
reads the active timed section at each 0.1 s sample and targets its endpoint;
wait sections command zero speed. At the goal, the same endpoint tracker is
used. A final partial integration step is allowed. This is a prescribed
tracking controller, not the native nominal straight-segment motion model.
Contacts caused by that model mismatch are not nominal CCBS planning failures.

## Required verification

- Binary MIP row constraints, reciprocity, and integrality checked on every new
  assignment. Objective checked against exhaustive enumeration for every one
  of the 64 possible directed conflict patterns with three robots.
- Independent swept-segment checker validates CCBS paths, speed, endpoints,
  adjacency, and free grid vertices. Tests include between-sample collisions
  and fractional waits.
- Existing controller tests remain in the suite. New tests cover the baseline
  assignment and CCBS adapter. No test asserts experimental superiority.
- Summary script checks cache hashes, seed coverage, and original controller
  count diagnostics before emitting results. No invalid/failed planning run
  is counted as collision-free.
- Paired 95% percentile bootstrap intervals use 10,000 resamples with seed 913.
  They describe the sampled warehouse instances, not other domains or a
  safety certificate; threshold checks are reported together.
- Scripts, model table, inputs, logs, episode outcomes, toolchain versions,
  and SHA-256 provenance are retained under mapf/.

## Manuscript integration

All additions in this revision use `\suggest{}`. The matched-flee definition
belongs immediately before the lifelong result table. The MIP adaptation and
separate CCBS execution table follow it. The conditional proposition uses
barrier enclosure and continuous enforcement, not an unsupported numerical HJ
certificate. Results identify the filter's goal-progress tradeoff and retain
infeasibility/fallback limitations. Historical displacement-noise sweep numbers
are not evidence for the new simultaneous controller.
