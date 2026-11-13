"""Generate runtime, peak-memory, sample, depth, and dimension scaling records."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import tracemalloc

import numpy as np

from picard import NestedPicardSolver, constant_coefficient


def analytic_problem(dimension: int, *, amplitude: float = 0.1, delta: float = 0.4):
    """Bounded advection--diffusion problem with a nonlinear transformed source."""
    velocity = np.linspace(0.05, 0.15, dimension) / np.sqrt(dimension)

    def initial(states):
        states = np.asarray(states)
        return amplitude * np.mean(np.cos(states), axis=-1)

    def hamiltonian(time, state, gradient):
        del time, state
        return float(np.dot(velocity, gradient))

    def truth(time, state):
        phase = state - velocity * time
        value = amplitude * np.exp(-0.5 * delta * time) * np.mean(np.cos(phase))
        gradient = -amplitude * np.exp(-0.5 * delta * time) * np.sin(phase) / dimension
        return float(value), gradient

    return initial, hamiltonian, truth


def current_rss_bytes() -> int:
    """Read the resident set size from Linux procfs when available."""
    try:
        fields = Path("/proc/self/status").read_text().splitlines()
        value = next(line for line in fields if line.startswith("VmRSS:"))
        return int(value.split()[1]) * 1024
    except (OSError, StopIteration, ValueError):
        return 0


def run_case(dimension: int, samples: int, level: int, seed: int, horizon: float) -> dict:
    initial, hamiltonian, truth = analytic_problem(dimension)
    solver = NestedPicardSolver(
        initial,
        constant_coefficient(0.7),
        hamiltonian,
        delta=0.4,
        samples=samples,
    )
    state = np.zeros(dimension)
    rss_before = current_rss_bytes()
    tracemalloc.start()
    start = time.perf_counter()
    result = solver.evaluate(level, horizon, state, seed=seed)
    runtime = time.perf_counter() - start
    _, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    expected, expected_gradient = truth(horizon, state)
    coordinate_payload = samples * dimension * np.dtype(float).itemsize
    # Each active nonlinear level retains source noise and shifted coordinates;
    # the base level retains one heat-kernel sample array.
    depth_linear_bound = (2 * level + 1) * coordinate_payload
    return dict(
        dimension=dimension,
        samples=samples,
        level=level,
        seed=seed,
        horizon=horizon,
        runtime_seconds=runtime,
        traced_peak_bytes=int(traced_peak),
        rss_before_bytes=int(rss_before),
        peak_rss_bytes=int(peak_rss),
        incremental_peak_rss_bytes=max(0, int(peak_rss) - int(rss_before)),
        coordinate_payload_bytes=int(coordinate_payload),
        depth_linear_coordinate_bound_bytes=int(depth_linear_bound),
        value=result.value,
        expected_value=expected,
        absolute_value_error=abs(result.value - expected),
        gradient_l2_error=float(np.linalg.norm(result.gradient - expected_gradient)),
        transformed_value=result.transformed_value,
        stats=result.stats,
    )


def worker(args) -> None:
    record = run_case(args.dimension, args.samples, args.level, args.seed, args.horizon)
    print(json.dumps(record, separators=(",", ":")))


def child_case(script: Path, **kwargs) -> dict:
    command = [
        sys.executable,
        str(script),
        "--worker",
        "--dimension", str(kwargs["dimension"]),
        "--samples", str(kwargs["samples"]),
        "--level", str(kwargs["level"]),
        "--seed", str(kwargs["seed"]),
        "--horizon", str(kwargs["horizon"]),
    ]
    process = subprocess.run(command, check=True, text=True, capture_output=True)
    return json.loads(process.stdout.strip().splitlines()[-1])


def median_record(records: list[dict]) -> dict:
    output = dict(records[0])
    for field in [
        "runtime_seconds", "traced_peak_bytes", "rss_before_bytes", "peak_rss_bytes",
        "incremental_peak_rss_bytes", "absolute_value_error", "gradient_l2_error",
        "value", "transformed_value",
    ]:
        output[field] = float(np.median([record[field] for record in records]))
    output["replicate_records"] = records
    return output


def slope(records: list[dict], x_field: str, y_field: str) -> float:
    x = np.asarray([record[x_field] for record in records], dtype=float)
    y = np.asarray([record[y_field] for record in records], dtype=float)
    mask = (x > 0) & (y > 0)
    if np.sum(mask) < 2:
        return float("nan")
    return float(np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)[0])


def run_suite(output: Path, *, replicates: int) -> dict:
    script = Path(__file__).resolve()
    horizon = 0.15
    plans = {
        "dimension": [dict(dimension=d, samples=12, level=2)
                      for d in [8, 16, 32, 64, 128, 256, 512, 1024, 2048]],
        "samples": [dict(dimension=8, samples=n, level=2)
                    for n in [4, 6, 8, 12, 16, 24, 32, 48]],
        "picard_level": [dict(dimension=8, samples=6, level=k) for k in [0, 1, 2, 3]],
    }
    sweeps = {}
    for name, plan in plans.items():
        rows = []
        for case_index, case in enumerate(plan):
            trials = [
                child_case(
                    script,
                    **case,
                    seed=20261113 + 1000 * case_index + replicate,
                    horizon=horizon,
                )
                for replicate in range(replicates)
            ]
            rows.append(median_record(trials))
            print(name, case, f"runtime={rows[-1]['runtime_seconds']:.4g}s", flush=True)
        sweeps[name] = rows

    summary = dict(
        schema_version=1,
        algorithm="depth-first nested nonlinear Monte Carlo Picard point evaluation",
        equation="viscous linear advection HJ equation; nonlinear Cole--Hopf source retained",
        horizon=horizon,
        replicates=replicates,
        environment=dict(
            python=sys.version,
            numpy=np.__version__,
            platform=platform.platform(),
            cpu_count=os.cpu_count(),
        ),
        source_sha256={
            name: hashlib.sha256((script.parent / name).read_bytes()).hexdigest()
            for name in ["picard.py", "scaling.py"]
        },
        slopes=dict(
            runtime_vs_dimension=slope(sweeps["dimension"], "dimension", "runtime_seconds"),
            traced_memory_vs_dimension=slope(sweeps["dimension"], "dimension", "traced_peak_bytes"),
            runtime_vs_samples=slope(sweeps["samples"], "samples", "runtime_seconds"),
            traced_memory_vs_samples=slope(sweeps["samples"], "samples", "traced_peak_bytes"),
        ),
        sweeps=sweeps,
        interpretation=(
            "The point evaluator avoids tensor-grid storage. Its depth-first live coordinate payload "
            "is linear in state dimension and sample count for fixed Picard depth. Runtime grows "
            "rapidly with sample count and Picard depth because the recursion is nested. These "
            "measurements support a point-query storage claim, not dimension-independent runtime, "
            "continuum reconstruction, or a population-wide safety certificate."
        ),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")

    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="") as stream:
        fields = [
            "sweep", "dimension", "samples", "level", "runtime_seconds",
            "traced_peak_bytes", "incremental_peak_rss_bytes",
            "depth_linear_coordinate_bound_bytes", "absolute_value_error",
            "gradient_l2_error", "gaussian_vectors", "source_evaluations",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for sweep_name, rows in sweeps.items():
            for row in rows:
                writer.writerow(dict(
                    sweep=sweep_name,
                    dimension=row["dimension"],
                    samples=row["samples"],
                    level=row["level"],
                    runtime_seconds=row["runtime_seconds"],
                    traced_peak_bytes=row["traced_peak_bytes"],
                    incremental_peak_rss_bytes=row["incremental_peak_rss_bytes"],
                    depth_linear_coordinate_bound_bytes=row["depth_linear_coordinate_bound_bytes"],
                    absolute_value_error=row["absolute_value_error"],
                    gradient_l2_error=row["gradient_l2_error"],
                    gaussian_vectors=row["stats"]["gaussian_vectors"],
                    source_evaluations=row["stats"]["source_evaluations"],
                ))
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/scaling.json"))
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dimension", type=int, default=8)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--horizon", type=float, default=0.15)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.worker:
        worker(arguments)
    else:
        if arguments.replicates < 1:
            raise SystemExit("--replicates must be positive")
        run_suite(arguments.output, replicates=arguments.replicates)
