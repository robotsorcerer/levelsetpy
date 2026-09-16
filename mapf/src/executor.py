"""Continuous Dubins/unicycle executor with bounded disturbance.

Rolls out the committed grid plans on TRUE dynamics: a fixed-speed unicycle with
bounded turn rate (so turn radius = v/omega_max is finite and matches the BRT
parameters). Additive bounded disturbance models localization/actuation error.
Realized collisions are detected in continuous space (center distance < 2r),
i.e. the ground truth that a dynamics-blind planner cannot see.

CRN: the disturbance generator is seeded per replication and draws a FIXED
number of values per (macro-step, agent, substep), so two policies run under the
same seed observe the identical noise realization (paired comparison).
"""

from __future__ import annotations
import time
import numpy as np

from hj_conflict import wrap_angle
from compatible_shield import CompatibleShield


class DubinsParams:
    def __init__(self, speed=1.0, omega_max=1.0, k_theta=4.0,
                 robot_radius=0.25, macro_T=1.0, substeps=10,
                 dist_sigma=0.05, dist_clip=0.12,
                 disturbance_mode="legacy_displacement"):
        if disturbance_mode not in ("legacy_displacement", "velocity"):
            raise ValueError("Unknown disturbance mode")
        if (not all(np.isfinite(v) and v > 0 for v in
                    (speed, omega_max, k_theta, robot_radius, macro_T))
                or not isinstance(substeps, int) or substeps < 1
                or not all(np.isfinite(v) and v >= 0 for v in (dist_sigma, dist_clip))):
            raise ValueError("Invalid executor parameters")
        self.speed = speed
        self.omega_max = omega_max
        self.k_theta = k_theta
        self.robot_radius = robot_radius
        self.macro_T = macro_T
        self.substeps = substeps
        self.dt = macro_T / substeps
        self.dist_sigma = dist_sigma
        self.dist_clip = dist_clip
        self.collision_dist = 2.0 * robot_radius
        self.disturbance_mode = disturbance_mode


def init_cpose(agents):
    """Continuous pose dict from discrete agent cells (x=col, y=row)."""
    return {a.id: np.array([float(a.cell[1]), float(a.cell[0]), a.theta])
            for a in agents}


def rollout(grid, agents, plans, cpose, params: DubinsParams,
            drng: np.random.Generator, exec_steps: int, shield=None,
            trace=None):
    """Advance `exec_steps` macro-steps on continuous dynamics.

    A membership predicate uses the historical first-threat flee controller.
    A CompatibleShield projects turn commands against simultaneous pairwise
    constraints. Neither sampled numerical mode is a fleet-safety certificate.
    Compatible steering requires bounded velocity disturbances; legacy
    displacement noise remains available to reproduce earlier experiments.

    Mutates `cpose` and each agent's discrete `cell`/`theta`. Returns dict with
    realized collision count + number of shield interventions over this tick.
    """
    ids = [a.id for a in agents]
    agent_by_id = {a.id: a for a in agents}
    prio = {a.id: a.priority for a in agents}
    p = params
    collisions = 0
    interventions = 0
    compatible = isinstance(shield, CompatibleShield)
    if compatible:
        if p.disturbance_mode != "velocity":
            raise ValueError("CompatibleShield requires disturbance_mode='velocity'")
        if (not np.isclose(p.omega_max, shield.turn_bound)
                or p.speed > shield.table.meta["speed"] + 1e-12
                or p.collision_dist > shield.table.capture_radius + 1e-12
                or p.dist_clip > shield.disturbance_bound + 1e-12):
            raise ValueError("Executor bounds exceed the compatible shield model")
    diagnostics = dict(shield_time_s=0.0, infeasible_agent_steps=0,
                       unsatisfied_agent_steps=0, replan_agent_steps=0,
                       multi_threat_agent_steps=0, compatible_multi_threat_steps=0,
                       outside_domain_agent_steps=0, negative_barrier_agent_steps=0)
    in_contact = {(ids[i], ids[j]): False
                  for i in range(len(ids)) for j in range(i + 1, len(ids))}
    if p.disturbance_mode == "velocity":
        in_contact = {key: np.linalg.norm(cpose[key[0]][:2] - cpose[key[1]][:2]) < p.collision_dist
                      for key in in_contact}

    def target_cell(aid, step):
        cells = plans[aid][0]
        idx = min(step + 1, len(cells) - 1)
        return cells[idx]

    for step in range(exec_steps):
        targets = {aid: grid.cell_to_xy(target_cell(aid, step)) for aid in ids}
        for _sub in range(p.substeps):
            old_poses = {aid: cpose[aid].copy() for aid in ids}
            decision = None
            command_turns, command_speeds = {}, {}
            shield_start = time.perf_counter()
            # --- shield: decide who yields, to whom, this micro-step ---------
            yield_to = {}   # aid -> pose of higher-priority threat to steer from
            if compatible:
                poses = np.array([cpose[aid] for aid in ids])
                offsets = np.array([targets[aid] for aid in ids]) - poses[:, :2]
                desired = np.arctan2(offsets[:, 1], offsets[:, 0])
                speeds = np.where(np.linalg.norm(offsets, axis=1) > 0.05, p.speed, 0.0)
                nominal = np.clip(p.k_theta * wrap_angle(desired - poses[:, 2]),
                                  -p.omega_max, p.omega_max)
                decision = shield.filter(poses, nominal, speeds, dt=p.dt)
                command_turns = dict(zip(ids, decision.turns))
                command_speeds = dict(zip(ids, speeds))
                changed = np.abs(decision.turns - nominal) > 1e-9
                yield_to = {aid: None for aid, modified in zip(ids, changed) if modified}
                interventions += int(changed.sum())
                multi = decision.active_pairs >= 2
                diagnostics["infeasible_agent_steps"] += int((~decision.interval_feasible).sum())
                diagnostics["unsatisfied_agent_steps"] += int((~decision.all_constraints_satisfied).sum())
                diagnostics["replan_agent_steps"] += int(decision.needs_replan.sum())
                diagnostics["multi_threat_agent_steps"] += int(multi.sum())
                diagnostics["compatible_multi_threat_steps"] += int(
                    (multi & decision.all_constraints_satisfied & (decision.min_barrier >= 0)).sum())
                diagnostics["outside_domain_agent_steps"] += int(decision.outside_domain.sum())
                diagnostics["negative_barrier_agent_steps"] += int((decision.min_barrier < 0).sum())
            elif shield is not None:
                for a_i in ids:
                    threat = None
                    for a_j in ids:
                        if a_i == a_j or prio[a_j] >= prio[a_i]:
                            continue  # only yield to strictly higher priority
                        pi, pj = cpose[a_i], cpose[a_j]
                        if shield.in_conflict(pj, pi) or shield.in_conflict(pi, pj):
                            threat = pj
                            break
                    if threat is not None:
                        yield_to[a_i] = threat
                        interventions += 1
            diagnostics["shield_time_s"] += time.perf_counter() - shield_start
            # --- integrate every agent one micro-step (fixed order = CRN) ----
            for aid in ids:
                x, y, th = cpose[aid]
                if compatible:
                    v = command_speeds[aid]
                    omega = command_turns[aid]
                elif aid in yield_to:
                    # Historical heuristic: FLEE away from the threat at speed
                    # (braking in place only spins the agent and gets it hit).
                    thx, thy, _ = yield_to[aid]
                    th_des = np.arctan2(y - thy, x - thx)   # away from threat
                    v = p.speed
                else:
                    tx, ty = targets[aid]
                    th_des = np.arctan2(ty - y, tx - x)
                    dist = np.hypot(tx - x, ty - y)
                    v = p.speed if dist > 0.05 else 0.0
                if not compatible:
                    dth = wrap_angle(th_des - th)
                    omega = np.clip(p.k_theta * dth, -p.omega_max, p.omega_max)
                dxn = np.clip(drng.normal(0, p.dist_sigma), -p.dist_clip, p.dist_clip)
                dyn = np.clip(drng.normal(0, p.dist_sigma), -p.dist_clip, p.dist_clip)
                if p.disturbance_mode == "velocity":
                    dxn *= p.dt
                    dyn *= p.dt
                x = x + (v * np.cos(th)) * p.dt + dxn
                y = y + (v * np.sin(th)) * p.dt + dyn
                th = wrap_angle(th + omega * p.dt)
                cpose[aid] = np.array([x, y, th])
            # collision check (continuous); count only onsets (sep -> contact)
            contacts = []
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    a_i, a_j = ids[i], ids[j]
                    d = np.hypot(cpose[a_i][0] - cpose[a_j][0],
                                 cpose[a_i][1] - cpose[a_j][1])
                    key = (a_i, a_j)
                    crossed = d < p.collision_dist
                    if p.disturbance_mode == "velocity":
                        # Swept segments of this Euler step, including contacts
                        # that both robots have left before its endpoint.
                        start = old_poses[a_i][:2] - old_poses[a_j][:2]
                        end = cpose[a_i][:2] - cpose[a_j][:2]
                        delta = end - start
                        denom = float(delta @ delta)
                        fraction = np.clip(-float(start @ delta) / denom, 0, 1) if denom else 0
                        crossed = np.linalg.norm(start + fraction * delta) < p.collision_dist
                    if crossed:
                        contacts.append(key)
                        if not in_contact[key]:
                            collisions += 1
                    in_contact[key] = d < p.collision_dist
            if trace is not None:
                trace.append({
                    "poses": {aid: cpose[aid].copy() for aid in ids},
                    "goals": {aid: agent_by_id[aid].goal for aid in ids},
                    "yielding": set(yield_to.keys()),
                    "contacts": contacts,
                    "collisions": collisions,
                    "interventions": interventions,
                    "turns": command_turns.copy(),
                    "constraint_feasible": None if decision is None else decision.all_constraints_satisfied.copy(),
                    "active_pairs": None if decision is None else decision.active_pairs.copy(),
                    "needs_replan": None if decision is None else decision.needs_replan.copy(),
                })
        # discrete advance: snap planner state to the committed cell
        for aid in ids:
            tc = target_cell(aid, step)
            agent_by_id[aid].cell = tc
            agent_by_id[aid].theta = float(cpose[aid][2])

    return {"collisions": collisions, "interventions": interventions, **diagnostics}
