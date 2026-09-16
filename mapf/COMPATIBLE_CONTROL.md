# Simultaneous pairwise steering constraints

The new execution filter projects the nominal turn rate onto the intersection of the steering constraints for every nearby opponent. It can accommodate several threats with one command when their constraints are compatible, and it exposes an empty intersection to the caller. It runs over the existing windowed geometric planner; it does not implement CBS or recursive feasibility.

This supports a controller-composition claim. It does not establish unconditional fleet safety or superiority over Chen's full algorithm.

## Relation to Chen (2016)

Chen, Shih, and Tomlin use a mixed-integer program to assign pairwise avoidance responsibilities. Each vehicle avoids at most one other vehicle. Their three-vehicle theorem provides a guarantee under stated initial safety and assignment conditions; larger fleets are also evaluated experimentally. See [Chen et al., *Multi-Vehicle Collision Avoidance via Hamilton-Jacobi Reachability and Mixed Integer Programming*](https://arxiv.org/abs/1603.05200).

There is no counterexample to the statement that safety of every pair implies fleet safety. The issue is whether the controls achieving pairwise safety are compatible:

\[
 \forall j\ne i,\;\exists u_i^{(j)}\text{ satisfying pair }(i,j)
 \quad\not\Rightarrow\quad
 \exists u_i\;\forall j\ne i,\;u_i\text{ satisfies pair }(i,j).
\]

Our filter computes the common admissible set directly. A vehicle may satisfy several pairwise inequalities with one command when their intersection is nonempty. Requiring every directed pair to be robust to the other vehicle's steering is conservative: our feasible set need not contain the set managed successfully by Chen's responsibility assignment. No dominance claim follows.

`single_constraint` selects the opponent with the smallest barrier value. It is a mechanism ablation, **not Chen's MIP**. `flee_matched` applies the historical first-threat flee rule with the same new table and disturbance convention as the other comparison arms.

## Derivation

The execution model is

\[
 \dot p_i=v_i(\cos\theta_i,\sin\theta_i)^\top+d_i,
 \quad \dot\theta_i=u_i,\quad |u_i|\le\bar\omega,
 \quad d_i\in[-D,D]^2.
\]

The disturbance components are world-frame **velocities**, multiplied by the integration timestep. The historical default adds bounded displacement at every substep; it remains available for old experiments.

Express the other vehicle in the ego vehicle's body frame:

\[
 q_{ij}=(x,y,\phi),\quad (x,y)^\top=R(-\theta_i)(p_j-p_i),
 \quad\phi=\theta_j-\theta_i.
\]

Without disturbance, the relative dynamics are

\[
 \dot x=v_j\cos\phi-v_i+u_i y,\qquad
 \dot y=v_j\sin\phi-u_i x,\qquad
 \dot\phi=u_j-u_i.
\]

The new cache uses these coordinates, with ego maximizing the safety value and the other vehicle minimizing it. The compatible controller rejects historical caches with a different dynamics/control convention. The NumPy solver uses a first-order Lax–Friedrichs scheme and periodic heading interpolation; it has no certified HJ approximation-error bound.

Let \(\widetilde V\) be the trilinear interpolant of the finite-horizon reachability table. Define

\[
 h_{ij}=\widetilde V(q_{ij})-\varepsilon_{\rm geom}-m,
 \qquad \varepsilon_{\rm geom}=\tfrac12\sqrt{\Delta x^2+\Delta y^2},
 \qquad m\ge0.
\]

At each control sample, require

\[
 \min_{|u_j|\le\bar\omega,\ d_i,d_j\in[-D,D]^2}
    \dot h_{ij}+\kappa h_{ij}\ge\rho,
 \quad\kappa>0,\quad\rho\ge0.
\]

For gradient \(g=(g_x,g_y,g_\phi)\), this becomes \(b_{ij}u_i+c_{ij}\ge0\), with

\[
\begin{aligned}
 b_{ij}&=g_xy-g_yx-g_\phi,\\
 c_{ij}&=g_x(v_j\cos\phi-v_i)+g_yv_j\sin\phi
          -\bar\omega|g_\phi|
          -2D\|R(\theta_i)(g_x,g_y)^\top\|_1
          +\kappa h_{ij}-\rho.
\end{aligned}
\]

The speed arguments are the commands used by the executor, including zero at a target. Commands are held over each integration step. Because the table was generated for fixed speeds, it serves as a candidate barrier when speeds vary, not as the value function of that changed game.

At an interpolation face, every incident-cell gradient supplies a constraint, including both sides of the heading seam. The common admissible set is

\[
 \mathcal U_i(q)=[-\bar\omega,\bar\omega]
       \cap\bigcap_{j\ne i}\{u_i:b_{ij}u_i+c_{ij}\ge0\},
\]

also intersected over gradient branches. Scalar steering makes this set an interval. If nonempty, clipping the nominal command into it solves

\[
 u_i^*=\arg\min_{u\in\mathcal U_i}(u-u_i^{\rm nom})^2.
\]

Zero-slope constraints are retained: a negative constant makes the intersection empty. Research mode then returns a bounded command maximizing the minimum constraint residual. This is a **least-violation fallback, not a safe backup**.

## Conditional safety argument

**Geometric enclosure.** The loader checks the nodal condition
\(V(x_a,y_b,\phi_c)\le\sqrt{x_a^2+y_b^2}-r\).
With nonnegative interpolation weights, the distance function's Lipschitz property gives

\[
 \widetilde V(q)\le\|q_{xy}\|-r+\mathbb E\|X-q_{xy}\|
 \le\|q_{xy}\|-r+\tfrac12\sqrt{\Delta x^2+\Delta y^2}.
\]

Here \(X\) is a cell corner sampled with the interpolation weights, so
\(\mathbb E\|X-q_{xy}\|^2\le(\Delta x^2+\Delta y^2)/4\).
Therefore \(h_{ij}\ge0\) implies pairwise non-overlap. This enclosure concerns the implemented interpolant, independently of HJ solver accuracy. A certified implementation must also cover finite arithmetic.

**Proposition.** Suppose all relevant pairs initially satisfy \(h_{ij}\ge0\), their trajectories are absolutely continuous and remain within the table domain, and a bounded measurable shared control satisfies every branch inequality almost everywhere throughout execution. Suppose the commanded speeds and disturbance bounds match the model. Then

\[
 \dot h_{ij}\ge-\kappa h_{ij}\text{ a.e.}
 \quad\Longrightarrow\quad
 h_{ij}(t)\ge e^{-\kappa t}h_{ij}(0)\ge0.
\]

All pairs, and hence the fleet, remain non-overlapping. The interpolant is locally Lipschitz; requiring all incident gradients suffices for its derivative along trajectories at cell interfaces. This applies the standard barrier invariance argument to a common feasible control. See [Ames et al., *Control Barrier Functions: Theory and Applications*](https://arxiv.org/abs/1903.11199).

**Continuing feasibility is an assumption, not a result.** An intersection may become empty even when each pair separately admits a control. The code detects this; it does not prove that a higher-level planner can prevent it. A finite-horizon HJ table does not automatically make the common barrier set controlled invariant.

**Sampling remains a gap.** The implementation checks inequalities at a sample and holds commands for `dt`. A verified bound on residual loss between samples, including changing gradients, integration error, and numeric tolerances, would be needed to justify a sampled-data reserve `rate_margin`. Its default is zero. A positive reserve chosen without that analysis is not a certificate. A uniform HJ value-error bound alone would not bound gradient or hold-interval errors.

**Domain coverage remains a condition.** The compatible controller never clamps a spatial query outside the table. It screens distant pairs using

\[
 \|p_j-p_i\|>r+2(v_{\max}+\sqrt2D)\max(\Delta t,T_{\rm BRT}).
\]

This precludes contact over the screening period under the speed bound. It does not prove a safe later handover into the barrier region. Near out-of-domain states are reported as unsupported. The proposition assumes complete domain coverage.

## API and failure handling

```python
from compatible_brt import ReachabilityTable
from compatible_shield import CompatibleShield, InfeasibleSafetyConstraints

table = ReachabilityTable("cache/compatible_brt.npz", margin=0.1)
shield = CompatibleShield(table, disturbance_bound=0.12)  # strict by default
try:
    decision = shield.filter(poses, nominal_turns, speeds, dt=0.1)
except InfeasibleSafetyConstraints as failure:
    diagnostics = failure.decision
    # Hand off to a verified backup/planner; this package has no such backup.
    raise
```

`poses` is an N-by-3 array; turns and speeds are N-vectors. `interval_feasible` concerns the selected constraint set. `all_constraints_satisfied` evaluates the returned command against every opponent, including in the one-threat ablation. `needs_replan` also flags negative barrier values and unsupported queries. Satisfying a derivative inequality does not repair a state already outside the barrier set.

The executor computes all commands from one pose snapshot. It reports infeasible agent-steps, violations, negative barriers, and successful multiple-threat constraints. It requires `DubinsParams(disturbance_mode="velocity")` and checks actuator, radius, and disturbance bounds. The research harness explicitly sets `fallback="least_violation"` so failures are counted. `needs_replan` is a diagnostic; recovery planning is not implemented.

## Reproduce

Runtime and precomputation require NumPy. Tests require pytest and use SciPy as an independent linear-programming oracle. Matplotlib is needed only for the optional plot.

```bash
python3 experiments/precompute_compatible_brt.py \
  --n-xy 81 --n-theta 64 --relative-disturbance 0.33941125496954283
pytest -q
python3 experiments/compatible_control_demo.py --plot
python3 experiments/compare_shields.py \
  --seeds 30 --n-agents 14 --total-ticks 40 \
  --out results/compatible_comparison_30seeds.json
```

The precomputation disturbance is \(2\sqrt2(0.12)\), an enclosing Euclidean bound for the difference of two world-axis disturbance boxes. Online constraints use the box support function directly.

The selected three-robot example starts with all six directed barrier values positive and all common intervals nonempty. All-constraint steering satisfies the inequalities through the three-second numerical run; one-threat steering violates an unselected constraint. Both remain collision-free. See [the data](results/compatible_three_robot.json) and [the plot](results/compatible_three_robot.svg). This selected example demonstrates the mechanism, not statistical performance.

The warehouse comparison uses common random numbers, identical starts/tasks, the same new table for HJ arms, and bounded velocity errors in every arm. Collision counts are new contacts over swept straight segments of Euler integration, including crossings between sample endpoints. Initial contact is excluded from onset counts; continuing contact is not recounted at a replan. The historical experiments used displacement noise and endpoint detection, so their counts are not directly comparable.

Forty replanning iterations with three committed one-second steps are **120 simulated seconds**. The six-second planning window is independent of the **0.6-second BRT horizon**. Task completion still uses committed grid cells rather than physical arrival, and continuous obstacle constraints are absent. The new comparison omits throughput as an execution-performance outcome.

## Defensible introduction wording

> Pairwise safety does not by itself ensure that independently synthesized avoidance controls can be applied simultaneously. Chen et al. resolve this compatibility issue through higher-level assignment of pairwise avoidance responsibilities, with each vehicle assigned at most one opponent. We instead form the intersection of the steering inputs satisfying all active pairwise barrier conditions and project the nominal command onto this common admissible set. This permits one vehicle to satisfy several avoidance requirements concurrently whenever their intersection is nonempty, while explicitly identifying incompatible requirements. The resulting safety argument is conditional on continued control feasibility and satisfaction of the barrier conditions throughout execution.

For the current sampled code, the final sentence describes a conditional continuous-time argument. A certified runtime-shield claim requires the additional analysis above. Improved performance over Chen would also require evaluating Chen's actual assignment method under matched assumptions.
