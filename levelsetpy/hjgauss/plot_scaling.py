"""Render the scaling record without recomputing the benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/scaling.json"))
    parser.add_argument("--output", type=Path, default=Path("results/scaling.pdf"))
    args = parser.parse_args()
    record = json.loads(args.input.read_text())
    dimension = record["sweeps"]["dimension"]
    samples = record["sweeps"]["samples"]
    levels = record["sweeps"]["picard_level"]

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    axes[0].loglog([r["dimension"] for r in dimension],
                   [r["runtime_seconds"] for r in dimension], "o-")
    axes[0].set(xlabel="State dimension", ylabel="Runtime (s)", title="Dimension scaling")

    axes[1].loglog([r["dimension"] for r in dimension],
                   [r["traced_peak_bytes"] / 2**20 for r in dimension], "o-", label="Measured peak")
    axes[1].loglog([r["dimension"] for r in dimension],
                   [r["depth_linear_coordinate_bound_bytes"] / 2**20 for r in dimension],
                   "s--", label="Coordinate payload")
    axes[1].set(xlabel="State dimension", ylabel="Memory (MiB)", title="Memory scaling")
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].loglog([r["samples"] for r in samples],
                   [r["runtime_seconds"] for r in samples], "o-", label="Sample sweep")
    axes[2].plot([r["samples"] for r in levels],
                 [np.nan] * len(levels), alpha=0)
    twin = axes[2].twiny()
    twin.semilogy([r["level"] for r in levels],
                  [r["runtime_seconds"] for r in levels], "s--", color="tab:red", label="Picard depth")
    axes[2].set(xlabel="Samples per update", ylabel="Runtime (s)", title="Nested runtime")
    twin.set_xlabel("Picard level")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()
