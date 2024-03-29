__all__ = ['upwindFirstWENO5']

__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__credits__  	= "There are None."
__license__ 	= "MIT License"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"


import copy
import logging
import numpy as np
from levelsetpy.utilities import *
logger = logging.getLogger(__name__)

from .upwind_first_weno5a import upwindFirstWENO5a

def upwindFirstWENO5(grid, data, dim, generateAll =False):
    """
     upwindFirstWENO5: fifth order upwind approx of first derivative.

       [ derivL, derivR ] = upwindFirstWENO5(grid, data, dim, generateAll)

     Computes a fifth order directional approximation to the first derivative,
       using a Weighted Essentially Non-Oscillatory (WENO) approximation.

     The generateAll option is used for debugging, and possibly by
       higher order weighting schemes.  Under normal circumstances
       the default (generateAll = false) should be used.  Notice that
       the generateAll option will just return the three ENO approximations.

     parameters:
       grid	Grid structure (see processGrid.m for details).
       data        Data array.
       dim         Which dimension to compute derivative on.
       generateAll Return all possible third order upwind approximations.
                     If this boolean is true, then derivL and derivR will
                     be cell vectors containing all the approximations
                     instead of just the WENO approximation.  Note that
                     the ordering of these approximations may not be
                     consistent between upwindFirstWENO1 and upwindFirstWENO2.
                     (optional, default = 0)

       derivL      Left approximation of first derivative (same size as data).
       derivR      Right approximation of first derivative (same size as data).

     Lekan Aug 21, 2021
    """

    return upwindFirstWENO5a(grid, data, dim, generateAll)
