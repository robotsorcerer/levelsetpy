"""Full residual generalized Cole--Hopf research solvers."""
from .solver import Coefficient, inverse, inverse_gradient, residual, source, solve_periodic, gaussian_duhamel
__all__ = ["Coefficient", "inverse", "inverse_gradient", "residual", "source", "solve_periodic", "gaussian_duhamel"]
