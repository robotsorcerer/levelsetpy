from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

import pytest
from compatible_brt import ReachabilityTable, solve_brt


@pytest.fixture(scope="session")
def table_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("brt") / "table.npz"
    solve_brt(path, n_xy=41, n_theta=32)
    return path


@pytest.fixture(scope="session")
def table(table_path):
    return ReachabilityTable(table_path, margin=0.1)
