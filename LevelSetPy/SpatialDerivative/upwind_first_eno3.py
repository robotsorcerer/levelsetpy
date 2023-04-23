__all__ = ['upwindFirstENO3']

__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__credits__  	= "There are None."
__license__ 	= "Molux Licence"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"

import copy
import logging
import numpy as np
from LevelSetPy.Utilities import *
logger = logging.getLogger(__name__)


from .upwind_first_eno3a import upwindFirstENO3a

def  upwindFirstENO3(grid, data, dim, generateAll=0):
    """
     upwindFirstENO3: third order upwind approx of first derivative.

       [ derivL, derivR ] = upwindFirstENO3(grid, data, dim, generateAll)

     Computes a third order directional approximation to the first
       derivative, using an Essentially Non-Oscillatory (ENO) approximation.

     The generateAll option is used for debugging, and possibly by
       higher order weighting schemes.  Under normal circumstances
       the default (generateAll = false) should be used.

     parameters:
       grid	Grid structure (see processGrid.m for details).
       data        Data array.
       dim         Which dimension to compute derivative on.
       generateAll Return all possible third order upwind approximations.
                     If this boolean is true, then derivL and derivR will
                     be cell vectors containing all the approximations
                     instead of just the ENO approximation.
                     (optional, default = 0)

       derivL      Left approximation of first derivative (same size as data).
       derivR      Right approximation of first derivative (same size as data).

    Lekan on August 16, 2021
    """

    return upwindFirstENO3a(grid, data, dim, generateAll)
