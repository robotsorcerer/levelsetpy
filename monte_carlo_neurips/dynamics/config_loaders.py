# ============================================================================
#  Physical Constants Configuration
# ============================================================================
from pathlib import Path
from typing import NamedTuple

class RocketConfig(NamedTuple):
    """Physical constants for rocket dynamics (Section 4.1)."""
    A_THRUST: float        # Acceleration due to thrust (m/s²)
    GRAVITY: float         # Gravitational acceleration (m/s²)
    U_BOUND: float         # Control input bound (m/s²)
    CAPTURE_RADIUS: float  # Target capture radius (m)

def _load_rocket_config(config_file: str = "rockets.cfg") -> RocketConfig:
    """Load rocket configuration from .cfg file into NamedTuple."""
    config_path = Path(__file__).parent.parent / "config" / config_file

    # Execute config file in isolated namespace
    namespace = {}
    with open(config_path, 'r') as f:
        exec(f.read(), namespace)

    # Return as NamedTuple
    return RocketConfig(
        A_THRUST=namespace['A_THRUST'],
        GRAVITY=namespace['GRAVITY'],
        U_BOUND=namespace['U_BOUND'],
        CAPTURE_RADIUS=namespace['CAPTURE_RADIUS']
    )