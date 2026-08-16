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
import numpy as np

from hj_conflict import wrap_angle


class DubinsParams:
    def __init__(self, speed=1.0, omega_max=1.0, k_theta=4.0,
                 robot_radius=0.25, macro_T=1.0, substeps=10,
                 dist_sigma=0.05, dist_clip=0.12):
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


def init_cpose(agents):
    """Continuous pose dict from discrete agent cells (x=col, y=row)."""
    return {a.id: np.array([float(a.cell[1]), float(a.cell[0]), a.theta])
            for a in agents}


def rollout(grid, agents, plans, cpose, params: DubinsParams,
            drng: np.random.Generator, exec_steps: int, shield=None):
    """Advance `exec_steps` macro-steps on continuous dynamics.

    If `shield` (a BRTPredicate) is given, it actuates the certificate at
    runtime: for any pair whose relative Dubins state is inside the windowed
    BRT, the LOWER-priority agent (larger .priority) yields — it brakes and
    steers away from the higher-priority agent. This is the dynamics-aware
    certified avoidance (innovation #2); geometric policy passes shield=None.

    Mutates `cpose` and each agent's discrete `cell`/`theta`. Returns dict with
    realized collision count + number of shield interventions over this tick.
    """
    ids = [a.id for a in agents]
    agent_by_id = {a.id: a for a in agents}
    prio = {a.id: a.priority for a in agents}
    p = params
    collisions = 0
    interventions = 0
    in_contact = {(ids[i], ids[j]): False
                  for i in range(len(ids)) for j in range(i + 1, len(ids))}

    def target_cell(aid, step):
        cells = plans[aid][0]
        idx = min(step + 1, len(cells) - 1)
        return cells[idx]

    for step in range(exec_steps):
        targets = {aid: grid.cell_to_xy(target_cell(aid, step)) for aid in ids}
        for _sub in range(p.substeps):
            # --- shield: decide who yields, to whom, this micro-step ---------
            yield_to = {}   # aid -> pose of higher-priority threat to steer from
            if shield is not None:
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
            # --- integrate every agent one micro-step (fixed order = CRN) ----
            for aid in ids:
                x, y, th = cpose[aid]
                if aid in yield_to:
                    # certified avoidance: FLEE away from the threat at speed
                    # (braking in place only spins the agent and gets it hit).
                    thx, thy, _ = yield_to[aid]
                    th_des = np.arctan2(y - thy, x - thx)   # away from threat
                    v = p.speed
                else:
                    tx, ty = targets[aid]
                    th_des = np.arctan2(ty - y, tx - x)
                    dist = np.hypot(tx - x, ty - y)
                    v = p.speed if dist > 0.05 else 0.0
                dth = wrap_angle(th_des - th)
                omega = np.clip(p.k_theta * dth, -p.omega_max, p.omega_max)
                dxn = np.clip(drng.normal(0, p.dist_sigma), -p.dist_clip, p.dist_clip)
                dyn = np.clip(drng.normal(0, p.dist_sigma), -p.dist_clip, p.dist_clip)
                x = x + (v * np.cos(th)) * p.dt + dxn
                y = y + (v * np.sin(th)) * p.dt + dyn
                th = wrap_angle(th + omega * p.dt)
                cpose[aid] = np.array([x, y, th])
            # collision check (continuous); count only onsets (sep -> contact)
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    a_i, a_j = ids[i], ids[j]
                    d = np.hypot(cpose[a_i][0] - cpose[a_j][0],
                                 cpose[a_i][1] - cpose[a_j][1])
                    key = (a_i, a_j)
                    if d < p.collision_dist:
                        if not in_contact[key]:
                            collisions += 1
                            in_contact[key] = True
                    else:
                        in_contact[key] = False
        # discrete advance: snap planner state to the committed cell
        for aid in ids:
            tc = target_cell(aid, step)
            agent_by_id[aid].cell = tc
            agent_by_id[aid].theta = float(cpose[aid][2])

    return {"collisions": collisions, "interventions": interventions}
