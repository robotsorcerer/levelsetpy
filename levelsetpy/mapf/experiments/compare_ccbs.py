#!/usr/bin/env python3
"""Authors' CCBS on MAPF_R snapshots, with paired unicycle execution.

This is a separate one-task experiment, not a replacement for the lifelong
table. Unique goals are sampled because the authors' planner keeps agents at
their final goals. All controller arms track the SAME continuous-time schedule;
no integer rounding of waits or motion durations is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

import numpy as np
from exp_A_pipeline import make_grid
from compatible_brt import ReachabilityTable
from compatible_shield import CompatibleShield
from chen_mip import ChenMIPShield
from hj_conflict import wrap_angle


def position(path, t):
    for t0, t1, start, end in path:
        if t <= t1:
            f = np.clip((t - t0) / (t1 - t0), 0, 1)
            return (1 - f) * np.array(start) + f * np.array(end)
    return np.array(path[-1][3])


def swept_dist(start, end):
    delta = end - start
    denominator = np.sum(delta * delta, axis=-1)
    fraction = np.divide(-np.sum(start * delta, axis=-1), denominator,
                         out=np.zeros_like(denominator), where=denominator > 0)
    return np.linalg.norm(start + np.clip(fraction, 0, 1)[..., None] * delta, axis=-1)


def verify_plan(paths, starts, goals, grid):
    for i, path in enumerate(paths):
        assert np.allclose(path[0][2], starts[i][::-1])
        assert np.allclose(path[-1][3], goals[i][::-1])
        for k, (t0, t1, start, end) in enumerate(path):
            assert t1 > t0 and grid.is_free(tuple(int(x) for x in reversed(start))) and grid.is_free(tuple(int(x) for x in reversed(end)))
            distance = np.linalg.norm(np.array(start) - end)
            assert distance == 0 or abs(distance - 1) < 1e-7
            assert distance == 0 or abs(t1 - t0 - distance) < 1e-6
            if k:
                assert np.allclose(path[k - 1][3], start)
                assert abs(path[k - 1][1] - t0) < 1e-7
    minimum = float('inf')
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            times = sorted(set([0.] + [s[1] for p in (paths[i], paths[j]) for s in p]))
            for t0, t1 in zip(times[:-1], times[1:]):
                minimum = min(minimum, float(swept_dist(position(paths[i], t0) - position(paths[j], t0),
                                                        position(paths[i], t1) - position(paths[j], t1))))
    if minimum < .5 - 1e-6:
        raise AssertionError(f'CCBS nominal plan has collision: minimum distance {minimum}')
    return minimum


def execute(paths, table, seed, mode, duration, dt=.1):
    poses = np.array([[*path[0][2], 0.] for path in paths], dtype=float)
    n = len(poses)
    rng = np.random.default_rng(seed + 123)
    noise = np.clip(rng.normal(0, .05, (int(np.ceil(duration / dt)), n, 2)), -.12, .12)
    shield = None
    if mode == 'simultaneous':
        shield = CompatibleShield(table, disturbance_bound=.12, fallback='least_violation')
    elif mode == 'chen_mip':
        shield = ChenMIPShield(table, disturbance_bound=.12, fallback='least_violation', threshold=.15)
    ii, jj = np.triu_indices(n, 1)
    contact = np.linalg.norm(poses[ii, :2] - poses[jj, :2], axis=1) < .5
    contacts, minimum, infeasible = 0, float('inf'), 0
    controller_time = 0.
    for step, perturbation in enumerate(noise):
        t = step * dt
        step_dt = min(dt, duration - t)
        targets, waits = [], []
        for path in paths:
            segment = next((s for s in path if t < s[1] - 1e-10), path[-1])
            targets.append(segment[3])
            waits.append(segment[2] == segment[3])
        offsets = np.array(targets) - poses[:, :2]
        nominal = np.clip(4 * wrap_angle(np.arctan2(offsets[:, 1], offsets[:, 0]) - poses[:, 2]), -1, 1)
        speeds = np.where((np.linalg.norm(offsets, axis=1) > .05) & ~np.array(waits), 1., 0.)
        turns = nominal
        if shield is not None:
            tic = time.perf_counter()
            decision = shield.filter(poses, nominal, speeds, dt=step_dt)
            controller_time += time.perf_counter() - tic
            turns = decision.turns
            infeasible += int((~decision.interval_feasible).sum())
        old = poses.copy()
        poses[:, :2] += step_dt * (speeds[:, None] * np.c_[np.cos(poses[:, 2]), np.sin(poses[:, 2])] + perturbation)
        poses[:, 2] = wrap_angle(poses[:, 2] + turns * step_dt)
        distances = swept_dist(old[ii, :2] - old[jj, :2], poses[ii, :2] - poses[jj, :2])
        contacts += int(np.count_nonzero((distances < .5) & ~contact))
        contact = np.linalg.norm(poses[ii, :2] - poses[jj, :2], axis=1) < .5
        minimum = min(minimum, float(distances.min()))
    return dict(contacts=contacts, min_swept_separation=minimum,
                common_interval_infeasible_agent_steps=infeasible,
                control_seconds=controller_time, steps=len(noise),
                final_mean_goal_error=float(np.linalg.norm(poses[:, :2] - np.array([p[-1][3] for p in paths]), axis=1).mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seeds', type=int, default=30)
    ap.add_argument('--agents', type=int, default=14)
    ap.add_argument('--time-limit', type=float, default=10)
    ap.add_argument('--brt', type=Path, default=Path(__file__).resolve().parents[1] / 'cache/compatible_brt.npz')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    grid = make_grid()
    root = ET.Element('root'); node = ET.SubElement(root, 'map')
    ET.SubElement(node, 'width').text = str(grid.W)
    ET.SubElement(node, 'height').text = str(grid.H)
    cells = ET.SubElement(node, 'grid')
    for row in range(grid.H):
        ET.SubElement(cells, 'row').text = ' '.join(str(int(not grid.is_free((row, col)))) for col in range(grid.W))
    ET.ElementTree(root).write(args.out / 'map.xml')
    root = ET.Element('root'); node = ET.SubElement(root, 'algorithm')
    for key, value in dict(use_cardinal='true', use_disjoint_splitting='true', hlh_type=2,
                           connectedness=2, focal_weight=1., agent_size=.25,
                           timelimit=args.time_limit, precision=1e-7).items():
        ET.SubElement(node, key).text = str(value)
    ET.ElementTree(root).write(args.out / 'config.xml')
    table = ReachabilityTable(args.brt, margin=.1)
    rows = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        free = list(grid.free_cells); rng.shuffle(free)
        starts = [tuple(map(int, cell)) for cell in free[:args.agents]]
        endpoints = list(grid.endpoints)
        while True:
            order = rng.permutation(len(endpoints))[:args.agents]
            goals = [endpoints[k] for k in order]
            if all(a != b for a, b in zip(starts, goals)):
                break
        root = ET.Element('root')
        for start, goal in zip(starts, goals):
            ET.SubElement(root, 'agent', start_i=str(start[0]), start_j=str(start[1]),
                          goal_i=str(goal[0]), goal_j=str(goal[1]))
        task = args.out / f'seed_{seed:02d}.xml'
        ET.ElementTree(root).write(task)
        tic = time.perf_counter()
        try:
            process = subprocess.run([str(args.source / 'build/CCBS'), str(args.out / 'map.xml'), str(task), str(args.out / 'config.xml')],
                                     capture_output=True, text=True, timeout=args.time_limit + 5)
            stdout = process.stdout + process.stderr
            found = process.returncode == 0 and 'Soulution found: true' in stdout
        except subprocess.TimeoutExpired:
            found = False; stdout = 'External timeout'
        (args.out / f'seed_{seed:02d}.log').write_text(stdout)
        row = dict(seed=seed, solved=found, process_seconds=time.perf_counter() - tic, starts=starts, goals=goals)
        if found:
            log = ET.parse(task.with_name(task.stem + '_log.xml')).getroot().find('log')
            paths = []
            for agent in log.findall('agent'):
                path, t = [], 0.
                for section in agent.find('path').findall('section'):
                    duration = float(section.attrib['duration'])
                    start = [float(section.attrib['start_j']), float(section.attrib['start_i'])]
                    end = [float(section.attrib['goal_j']), float(section.attrib['goal_i'])]
                    if duration > 0:
                        path.append((t, t + duration, start, end)); t += duration
                paths.append(path)
            assert len(paths) == args.agents
            row['nominal_min_separation'] = verify_plan(paths, starts, goals, grid)
            row['makespan'] = max(path[-1][1] for path in paths)
            row['duration'] = row['makespan'] + 5.
            row['execution'] = {mode: execute(paths, table, seed, mode, row['duration'])
                                for mode in ('unfiltered', 'chen_mip', 'simultaneous')}
        rows.append(row)
        output = dict(protocol='CCBS MAPF_R one-task snapshots; three paired tracking arms, makespan + 5 s',
                      source_commit=subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip(),
                      binary_sha256=hashlib.sha256((args.source/'build/CCBS').read_bytes()).hexdigest(),
                      table_sha256=hashlib.sha256(args.brt.read_bytes()).hexdigest(),
                      arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                      limitations=['Not lifelong episodes', 'No continuous obstacle filter', 'Chen assignment uses finite-horizon table'], records=rows)
        (args.out / 'results.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
        print(seed, 'solved', found, row.get('execution', {}), flush=True)


if __name__ == '__main__':
    main()
