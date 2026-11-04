"""HJ-Gauss generalized Cole--Hopf research solvers."""
from .core.transform import Coefficient, inverse, inverse_gradient, residual, source, solve_periodic, gaussian_duhamel
__all__ = ["Coefficient", "inverse", "inverse_gradient", "residual", "source", "solve_periodic", "gaussian_duhamel"]
