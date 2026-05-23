"""Core HJ-Gauss solver module."""

from .config import SolverConfig
from .hj_sampler import HJReachabilitySampler
from .heat_solver import mc_value_at_point, mc_value_batch, mc_gradient_at_point, mc_gradient_batch
from .transforms import cole_hopf_forward, cole_hopf_inverse
from .topology import brt_topology_signature, detect_phase_transitions, TopologyState

__all__ = [
    "SolverConfig",
    "HJReachabilitySampler",
    "mc_value_at_point",
    "mc_value_batch",
    "mc_gradient_at_point",
    "mc_gradient_batch",
    "cole_hopf_forward",
    "cole_hopf_inverse",
    "brt_topology_signature",
    "detect_phase_transitions",
    "TopologyState",
]
