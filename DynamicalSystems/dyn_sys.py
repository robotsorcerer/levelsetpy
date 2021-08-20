""" This file defines the base dynamical systems class. """
import abc
from utils import cell

class DynSys(object):
    "Dynamical Systems Superclass."
    __metaclass__ = abc.ABCMeta

    def __init__(self, params):
        # For bookkeepping and plotting
        for var, val in params.items():
            object.__setattr__(self, var, val)

        # see test_dubins.py

    @abc.abstractmethod
    def update_dynamics(self, x):
        """
        Evaluate a single step of the dynamics.
        Args:
            x: The state space.
        """
        raise NotImplementedError("Must be implemented in subclass.")

    @abc.abstractmethod
    def get_opt_u(self, deriv, uMode='min', t=None, y = None):
        """
        Derive Optimal Control Law
        Args:
            deriv: derivative
            t: time
            y: measurement
        """
        raise NotImplementedError("Must be implemented in subclass.")

    @abc.abstractmethod
    def get_opt_v(self, deriv, uMode='min', t=None, y = None):
        """
        Derive Optimal Disturbance Law
        Args:
            deriv: derivative
            t: time
            y: measurement
        """
        raise NotImplementedError("Must be implemented in subclass.")

    @abc.abstractmethod
    def update_state(self,  u, T, x0, d):
        """
        Update state based on new control
        Args:
            % Inputs:   self - current dynamical systems object
                        u   - control (defaults to previous control)
                        T   - duration to hold control
                        x0  - initial state (defaults to current state if set to [])
                        d   - disturbance (defaults to [])

            Outputs:  x1  - final state

        """
        raise NotImplementedError("Must be implemented in subclass.")
