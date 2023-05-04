__all__ = ["DoubleIntegrator"]

__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__credits__  	= "There are None."
__license__ 	= "MIT License"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"
__date__ = "Nov. 22, 2021"

import cupy as cp
import numpy as np

class DoubleIntegrator():
    def __init__(self, grid, u_bound=1):
        """
            The base function for the double integrator's
            minimum time to reach problem.

            Dynamics: \ddot{x}= u,  u \in [-1,1]

            This can represent a car with position
            x \in \mathbb{R} and with bounded acceleration u acting
            as the control (negative acceleration corresponds to braking).
            Let us study the problem of ``parking" the car at the origin,
            i.e., bringing it to rest at $ x=0$ , in minimal time.
            It is clear that the system can indeed be brought to rest
            at the origin from every initial condition. However, since the
            control is bounded, we cannot do this arbitrarily fast (we are
            ignoring the trivial case when the system is initialized at the
            origin). Thus we expect that there exists an optimal control
            u^* which achieves the transfer in the smallest amount of time.

            Ref: http://liberzon.csl.illinois.edu/teaching/cvoc/node85.html

            Parameters
            ==========
                grid: an np.meshgrid state space on which we are resolving this integrator dynamics.
        """

        self.grid     = grid
        self.control_law = u_bound
        self.Gamma = self.switching_curve # switching curve

    @property
    def switching_curve(self):
        """
            \Gamma = -(1/2) . x_2 . |x_2|
        """
        self.Gamma = -.5*self.grid.xs[1]*np.abs(self.grid.xs[1])

        return self.Gamma

    def hamiltonian(self, t, data, value_derivs, finite_diff_bundle):
        """
            H = \dot{x1} . x2 + \dot{x2} . u + x_0

            Here, x_0 is initial state which is zero since we are not interested in the
            running cost.

            Parameters
            ==========
            value: Value function at this time step, t
            value_derivs: Spatial derivatives (finite difference) of
                        value function's grid points computed with
                        upwinding.
            finite_diff_bundle: Bundle for finite difference function
                .innerData: Bundle with the following fields:
                    .partialFunc: RHS of the o.d.e of the system under consideration
                        (see function dynamics below for its impl).
                    .hamFunc: Hamiltonian (this function).
                    .dissFunc: artificial dissipation function.
                    .derivFunc: Upwinding scheme (upwindFirstENO2).
                    .innerFunc: terminal Lax Friedrichs integration scheme.
        """
        x2 = cp.asarray(self.grid.xs[1])

        return -(value_derivs[0]*x2- \
                 cp.abs(value_derivs[1])*self.control_law)

    def dissipation(self, t, data, derivMin, derivMax, \
                      schemeData, dim):
        """
            Parameters
            ==========
                dim: The dissipation of the Hamiltonian on
                the grid (see 5.11-5.12 of O&F).
        """
        x_dot = [
                    cp.asarray(np.abs(self.grid.xs[1])),
                    cp.abs(self.control_law)
        ]

        return x_dot[dim]

    def mttr(self):
        """
            Computes the minimum time we need to reach the
            switching curve:

            x2 + (sqrt(4x_1 + 2 x_2^2).(x_1 > \Gamma)) +
            (-x_2 + sqrt(2x_2^2 - 4 x_1) . (x_1 < \Gamma) +
            (|x_2| . (x_1 == \Gamma)).
        """

        #be sure to update the switching curve first
        self.switching_curve

        #  Compute the current state on or outside of the
        # switching curve.

        above_curve = (self.grid.xs[0]>self.Gamma)
        below_curve = (self.grid.xs[0]<self.Gamma)
        on_curve    = (self.grid.xs[0]==self.Gamma)

        reach_term1  = (self.grid.xs[1] + np.emath.sqrt(4*self.grid.xs[0] + \
                         2 * self.grid.xs[1]**2))*above_curve
        reach_term2 =  (-self.grid.xs[1]+np.emath.sqrt(-4*self.grid.xs[0] + \
                        2 * self.grid.xs[1]**2) )*below_curve
        reach_term3 = np.abs(self.grid.xs[1]) * on_curve

        reach_time = reach_term1 + reach_term2 + reach_term3

        return reach_time.real
