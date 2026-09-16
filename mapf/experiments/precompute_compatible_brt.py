#!/usr/bin/env python3
"""Build a new cache consistent with the executor's ego-body coordinates."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from compatible_brt import solve_brt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "cache" / "compatible_brt.npz"))
    parser.add_argument("--horizon", type=float, default=0.6)
    parser.add_argument("--capture-radius", type=float, default=0.5)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--turn-rate", type=float, default=1.0)
    parser.add_argument("--relative-disturbance", type=float, default=0.0)
    parser.add_argument("--x-lim", type=float, default=4.0)
    parser.add_argument("--n-xy", type=int, default=61)
    parser.add_argument("--n-theta", type=int, default=48)
    args = vars(parser.parse_args())
    path = args.pop("out")
    print(json.dumps(solve_brt(path, **args), indent=2))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
