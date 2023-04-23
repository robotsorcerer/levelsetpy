__all__ = ["termSum"]

__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__credits__  	= "There are None."
__license__ 	= "Molux Licence"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"

import numpy as np
from LevelSetPy.Utilities import *

def termSum(t, y, schemeData):
    """
     termSum: Combine a collection of spatial HJ term approximations.

     [ ydot, stepBound, schemeData ] = termSum(t, y, schemeData)

     This function independently evaluates a collection of HJ term
       approximations and returns their elementwise sum.

     Note that the HJ term approximations must be completely independent
       of one another.

     The CFL restrictions are inverse summed:
    	stepBound_sum = (\sum_i (1 / stepBound_i))^-1

     parameters:
       t            Time at beginning of timestep.
       y            Data array in vector form.
       schemeData	 A structure (see below).

       ydot	 Change in the data array, in vector form.
       stepBound	 CFL bound on timestep for stability.
       schemeData   A structure (see below).

     schemeData is a structure containing data specific to this type of
       term approximation.  For this function it contains the field(s)

       .innerFunc   A cell vector of function handles to the term
                      approximations that will be summed.
       .innerData   A cell vector of schemeData structures to pass to
                      each of the innerFunc elements.  This vector
                      must be the same size as the vector innerFunc.

     It may contain addition fields at the user's discretion.

     While termSum will not change the schemeData structure,
       the schemeData.innerData components may be changed during the calls
       to the schemeData.innerFunc function handles.

     For evolving vector level sets, y may be a cell vector.  In this case
       the entire y cell vector is passed unchanged in the calls to the
       schemeData.innerFunc function handles.

     If y is a cell vector, schemeData may be a cell vector of equal len.
       In this case, schemeData[0] contains the fields listed above.  In the
       calls to the schemeData[0].innerFunc function handles, the schemeData
       cell vector is passed unchanged except that the element schemeData[0]
       is replaced with the corresponding element of the
       schemeData[0].innerData cell vector.

      For vector level sets, get the first element.

   Lekan Molu, 08/21/21
  """
    if iscell(schemeData):
        thisSchemeData = schemeData[0]
    else:
        thisSchemeData = schemeData

    assert isfield(thisSchemeData, 'innerFunc'), "innerFunc not in schemeData"

    #Check that innerFunc and innerData are the same size cell vectors.
    if(not iscell(thisSchemeData.innerFunc) or not iscell(thisSchemeData.innerData)):
        error('schemeData.innerFunc and schemeData.innerData must be cell vectors')

    numSchemes = len(thisSchemeData.innerFunc.flatten())

    if(numSchemes != len(thisSchemeData.innerData.flatten())):
        error('schemeData.innerFunc and schemeData.innerData must be the same len')

    #Calculate sum of updates (inverse sum of stepBounds).
    ydot = 0
    stepBoundInv = 0
    for i in range(numSchemes):
        #Extract the appropriate inner data structure.
        if(iscell(schemeData)):
            innerData = schemeData
            innerData[0] = schemeData[0].innerData[i]
        else:
            innerData = schemeData.innerData[i]

    # Compute this component of the update.
    updateI, stepBoundI, innerData = thisSchemeData.innerFunc[i](t, y, innerData)
    ydot += updateI
    stepBoundInv += (1 / stepBoundI)

    # Store any modifications of the inner data structure.
    if(iscell(schemeData)):
        schemeData[0].innerData[i] = innerData[0]
    else:
        schemeData.innerData[i] = innerData

    # Final timestep bound.
    if(stepBoundInv == 0):
        stepBound = np.inf
    else:
        stepBound = 1 / stepBoundInv

    return ydot, stepBound, schemeData
