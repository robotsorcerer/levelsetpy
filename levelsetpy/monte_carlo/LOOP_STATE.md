# Active multidimensional validation loop

Status: active; numerical/editorial acceptance has NOT been declared.

## Frozen numerical producers

- `full_residual_nd.py`: SHA-256 `561dd5b37d6270e3fb3d3d341285816fa6acb0b224f16c5e45a617d9d06f42ed`
- `multidimensional_validation.py`: SHA-256 `dbd3223f7088a933dcdb51e71836c8ce0abfa81f6159fdc786b496a8d3d5d0e2`

Do not change these while native jobs run. Their digests bind all run metadata. Report/render/scheduler changes are independent. Use `/usr/bin/python3.10`; the default Conda Python lacks pytest. The regression suite now has 30 tests (including independent four-dimensional control reduction).

## Evidence and fixed gates

`results/multidimensional/acceptance_criteria.json` fixes the thresholds. Do not relax them to accept runs. `selection.json` identifies current selected resolutions and time steps. `validation_report.py --selection ...` assesses all gates, including both manufactured methods/dimensions and a fresh 100,000-query population bound to selected native archive hashes.

- Both 2-D and 3-D manufactured solvers meet the exact-error gate. Finest transformed 3-D error is about 3e-8.
- Double integrator (128^2) and single Dubins (128^2 x 64) pass all numerical checks.
- Rockets 160^2 x 80, 400 steps: defect sup 0.1433207, refinement discrepancy 0.007481, independent discrepancy 0.003223. Three sensitivity jobs remain in progress.
- Relative Dubins 192^2 x 96, 400 steps: defect sup 0.1561277 FAILS the fixed 0.15 gate. Refinement discrepancy 0.009912 passes. A 224^2 x 112, 400-step full/reference pair is running; its medium is 192/400. Do not claim the 192 run passed.
- Murmuration 128^2 x 64, 800 steps: defect sup 0.282494 FAILS, RMS 0.011486 passes. The 192^2 x 96 full/reference pair is running. Supremum plateau was independently investigated: unresolved source cutoff error dominates; keep the same mask and thresholds. Refine both planar and angular resolution if needed.
- A first 100,000-state evaluation using the 128 bird fields is saved. All 100,000 pairs are finite, seed 20261101; positive count 99828, reference 99811, 17 paired sign disagreements. These are descriptive query signs, not certificates. Re-evaluate on the final accepted bird fields; current population is deliberately ineligible for the selected 192 run.

## Running jobs

Inspect `ps -eo pid,etime,pcpu,rss,args` and these logs before launching anything twice:

- `/tmp/hj_refinement_round2.log`: rockets 160 full/reference completed; relative 192 full completed; relative reference and bird 192 full/reference remaining.
- `/tmp/hj_rockets_sensitivity2.log`: rockets 160 time-halving, coefficient, combined box.
- `/tmp/hj_relative_round3.log`: relative Dubins 224 full/reference.

Initial 128-grid sensitivity jobs for superseded grids were canceled after the single-Dubins sensitivities completed. Their completed records remain valid. `superseded_batch.json` records this; do not restart obsolete jobs. Do not overcommit RAM; the machine has 31 GiB with other workloads and full swap.

## Manuscript integration already performed

Active document is `/home/lex/Documents/Papers/MoluxLabs/HJ_Gauss/royal_soc.tex`, which inputs `sec_transfull/*.tex`.

- Figure 1 now uses `arxiv/figures/rockets_3d_slices_fullsource.png` (new canonical 128 full/reference data, finite-grid diagnostic while refinement continues). Updated caption identifies terminal-horizon values, matched viscosity, dimensionless units. Replace it with the final accepted 160 run after all gates pass.
- The current `royal_soc.pdf` HAS been rebuilt and replaced, 37 pages. Its page 3 visibly contains the new Figure 1. Previous PDF backup: `/tmp/hj_previous_royal_soc.pdf`. Build directory: `/tmp/hj_full_residual_build`. No unresolved refs after second pass. Existing template front-matter/output-box warnings remain; rendered pages 1--3 inspected.
- Dubins coordinate and target/barrier illustrations were removed at the user's request, with `\cite{Mitchell2020}` (previous bibliography item 41). The rocket schematic (previous Fig. 10) was removed, with `\cite{MoluxLevPyCDC}`. Do not hardcode figure or bibliography numbers.
- The manufactured validation figure was moved to the appendix, retaining its discussion and adding exact 2-D/3-D verification prose. The user's later Fig. 2/3 removal with reference 41 was interpreted as the renumbered Dubins illustrations. Keep the manufactured figure in appendix per the agreed move.
- Appendix C now leads with relative Dubins and puts its new full/reference comparison alongside the model. Duplicate old single-Dubins/double-integrator plots removed; new figures carry both old and new labels. Normalized rocket model remains concise, with original physical-unit illustrations cited to Molu.
- Main results, abstract, introduction ending, conclusion, limitations rewritten around full residual numerical scope. The old endpoint algorithm/formulas were removed from active methods; section headings retained. Old 30-seed/45D endpoint tables and false threshold claims removed; joint 45-D dynamics retained as context, explicitly uncomputed by current full solver.
- `write_validation_tex.py` produces numeric macros and tables. Some rows currently say pending; this is an in-progress draft, NOT a final release. Regenerate after all data arrive. Never hand-edit generated numbers.
- Bird figures in appendix still point to OLD JPGs and must be replaced with newly rendered PNGs before completion. The optional old swept-tube figure was already commented by the user; keep it commented unless explicitly reinstated.
- `render_multidimensional_validation.py` supports selection files and case-gated rendering; controls already rendered from passing runs. New `render_population_validation.py` renders six bird slices, finite-window topology against reference, and 2,000 displayed points drawn from the saved complete 100,000 cohort. Run for final selected bird grid; regenerate at every selected-field change.
- Colorbar headings were moved above extension tips (18--20 source points). Figures use DejaVu Sans bold and sentence case. Actual TeX text/column width is 384.1122 pt = 384.1122/72.27 inches, NOT 6.3 inches; renderer metadata now uses the actual width. Rerender final assets once at that metadata setting. Minimum source text 11 pt on 8-inch canvas gives about 7.31 pt in the paper; axes are 14 pt source (~9.3 pt compiled).

## Remaining work

1. Finish the selected full/reference refinements. For failed gates, diagnose/refine while preserving fixed thresholds, domain mask, and physical model. Do not substitute smaller RMS for failed maximum.
2. Run sensitivity stages for final relative and bird resolutions once base gates pass. `refine_failed_cases.py` currently selects relative224, bird192, rocket160. The old 192 relative reference is useful medium/reference evidence even after selection224.
3. Run final `population_validation.py --n N --steps K`; verify 100,000 saved paired values and native hash bindings. Final population producer explicitly states a query cohort, not simulated coupled bird trajectories or safety certification.
4. Complete field archive sealing with `seal_validation.py` once jobs stop. Address reviewer cache/atomicity/failure-history findings in scheduler; do not alter frozen numerical producers merely for metadata cleanup. Keep successful convergence evidence, remove obsolete generated intermediates only after consumers are updated.
5. Render accepted figures into manuscript `arxiv/figures/`, update bird TeX paths/captions and topology interpretation; derive counts from actual fields (do not force expected topology). Coordinates/values are dimensionless. Paired topology may differ; disclose rather than certify.
6. Regenerate tables/macros and recomputation report, update remaining READMEs. `solver_nd.py`, `recompute_requested_figures.py`, and `render_cached_requested_panels.py` were removed because they retained obsolete filtering/model definitions; new producers replace them. The path inventory is updated.
7. Run final pytest, data integrity checks, source/caption/citation audit. Build with pdfLaTeX/BibTeX and two final pdfLaTeX passes in temporary output; copy final built PDF to `royal_soc.pdf` and visually inspect figures in the actual compiled pages and grayscale. Preserve generated font metadata and check actual placement sizes.
8. Finish independent Agentic-Writing review. 14 unique slots have completed (see below); a major revision requires 30. Use existing read-only reviewers, one slot per invocation, record real findings/dispositions and final digests. Never claim editorial acceptance merely from numerical thresholds. All blocking numerical and prose issues must be resolved before marking the goal complete.

## Independent review slots completed

Numerical reviewer `/root/numerical_review`: ND-MATH-01 (Nyquist bug, fixed), ND-MATH-02 (fix verified), ND-BOUNDARY-03 (combined sensitivity must be named; fixed in prose/report), ND-TESTS-04 (29 tests then; added independent 4D reduction test makes 30), ND-STABILITY-05 (unfiltered variants fail relative/rockets/birds; retain cutoff), ND-DEFECT-06 (max residual not yet converged in bird case; source cutoff error diagnosed).

Evidence reviewer `/root/evidence_review`: ND-PROVENANCE-01 (seal archives, freeze source, atomic cache, preserve failure history; pending final sealing/scheduler hardening), ND-METRICS-02 (overall gates missing manufactured/population; fixed), ND-METRICS-03 (bind population to native hashes and describe joint refinement; fixed), POP-PROVENANCE-04 (seeded cohort/producer pass; final fine values still pending).

Paper reviewer `/root/paper_review`: MODEL-PAPER-01 (tube sign, scalar/set target, DI Hamiltonian labels, time relation; fixed), MODEL-ROCKET-02 (separate normalized vector/rate model, false speed claim; fixed), MODEL-PAPER-03 (old analytic-barrier/minimum-time validation promises; qualified as context), CLAIM-SCOPE-04 (positivity/inversion wording, high-dimensional analytical-solution absolute, finite discretization versus exact viscosity solution; fixed).

All reviewers edited no files. Their complete final messages are in thread history. Future reviews must inspect updated frozen surfaces; do not count planned slots as completed.
