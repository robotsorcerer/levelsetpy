"""Monte Carlo HJ Reachability: Headline-grade 1M-bird aerial murmuration safety.

Main module exports for HJ-Gauss solver with 4D murmuration support.
Includes multi-GPU distribution with automatic single-GPU fallback.
"""

from .src.config import SolverConfig
from .src.hj_sampler import HJReachabilitySampler
from .src.gpu_distribution import GPUDistributor
from .src.hamiltonians.murmuration import MurmuationHamiltonian4D, MurmuationFlockHamiltonian4D
from .dynamics.murmuration_jax import (
    MurmuationSolverJAX4D,
    FlockState,
    PredatorState,
    terminal_cost_4d,
)
from .src.topology import TopologyState, brt_topology_signature, detect_phase_transitions

__all__ = [
    "SolverConfig",
    "HJReachabilitySampler",
    "GPUDistributor",
    "MurmuationHamiltonian4D",
    "MurmuationFlockHamiltonian4D",
    "MurmuationSolverJAX4D",
    "FlockState",
    "PredatorState",
    "terminal_cost_4d",
    "TopologyState",
    "brt_topology_signature",
    "detect_phase_transitions",
]
