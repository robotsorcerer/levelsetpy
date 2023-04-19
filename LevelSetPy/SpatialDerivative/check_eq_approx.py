__all__ = ['checkEquivalentApprox']

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

def checkEquivalentApprox(approx1, approx2,bound):
    """
     checkEquivalentApprox: Checks two derivative approximations for equivalence.

       [ relError, absError ] = checkEquivalentApprox(approx1, approx2, bound)

     Checks two derivative approximations for equivalence.

     A warning is generated if either of these conditions holds:
       1) The approximation magnitude > bound
          and the maximum relative error > bound.
       2) The approximation magnitude < bound
          and the maximum absolute error > bound.

     Normally, the return values are ignored
       (the whole point is the warning checks).

     parameters:
       approx1     An array containing one approximation.
       approx2     An array containing the other approximation.
       bound       The bound above which warnings are generated.

       relError    The relative error at each point in the array
                     where the magnitude > bound (NaN otherwise).
       absError    The absolute error at each point in the array.

    """

    # Approximate magnitude of the solution
    magnitude = 0.5 * np.abs(approx1 + approx2)

    # Which nodes deserve relative treatment, and which absolute treatment?
    useRelative = np.nonzero(magnitude > bound)
    useAbsolute = np.nonzero(magnitude <= bound)

    absError = np.abs(approx1 - approx2)

    # Be careful not to divide by too small a number.
    relError = ones(size(absError))
    relError.fill(np.nan)
    relError[useRelative] = np.divide(absError[useRelative], magnitude[useRelative])

    # Check that bounds are respected.
    if(max(relError[useRelative]) > bound):
        logger.warn(f'exceeded relative bound. Error in supposedly'
                    'equivalent derivative approximations'
                    '{max(relError[useRelative])}, {bound}')
    if(max(absError[useAbsolute]) > bound):
        logger.warn(f'exceeded absolute bound. Error in supposedly'
                    'equivalent derivative approximations'
                    '{max(relError[useAbsolute])}, {bound}')

    return relError, absError
