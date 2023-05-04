__all__ = ["postTimestepTTR"]

__author__ 		  = "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__credits__  	  = "There are None."
__license__ 	  = "MIT License"
__maintainer__ 	= "Lekan Molu"
__email__ 		  = "patlekno@icloud.com"
__status__ 		  = "Completed"
__date__        = "March 29, 2023"

import copy
import cupy as cp
import numpy as np 
from math import inf
from LevelSetPy.Utilities.matlab_utils import * 

def postTimestepTTR(t, yIn, schemeDataIn):
  """ 
    postTimestepTTR: PostTimestep routine to record Time To Reach (TTR).

    [ yOut, schemeDataOut ] = postTimestepTTR(t, yIn, schemeDataIn)

    This routine implements the postTimestepFunc prototype for a common
    operation: recording the time at which the zero level set crosses
    each node of the grid.

    The level set function itself is unmodified:  yOut = yIn.

    The time to reach function (and some auxiliary functions) are stored
    as members of the schemeData structure.  All the members listed
    below need not (and should not) be initialized by the user.

    The final time to reach function will be stored in schemeData.ttr 
    in vector form (which the user can reshape into array form if necessary).
    It will be set to infinity for those nodes which were never reached
    by the zero level set.

    Note that the minimum time reported by schemeData.ttr will be the first
    time at which postTimestepTTR is called.  The toolbox will normally
    call after the first timestep.  Consequently, if the user wants
    to record the initial conditions accurately, postTimestepTTR should 
    be called directly by the user with the initial data and initial time 
    before the toolbox time integration is started.

    The actual values stored in the time to reach are computed by linear
    interpolations of the crossing times between timesteps.

    Parameters
    ==========
    t              Current time.
    yIn            Input version of the level set function, in vector form.
    schemeDataIn   Input version of a structure (see below).

    yOut           Output version of the level set function, in vector form.
    schemeDataOut  Output version of the structure (unmodified).

    schemeData is a structure containing the time to reach function and
    some auxiliary functions.  This function ensures that the fields
    are properly initialized.

    .ttr           Time to reach function, in vector form.
    .ttrLastY      Data array at last timestep.
                        Auxiliary data that the user may ignore.
    .ttrLastT      Time at last timestep.
                        Auxiliary data that the user may ignore.

    Users should avoid conflicting names in the schemeData structure, but
    may include fields of other names.
  """

  schemeDataOut = copy.copy(schemeDataIn)
  yOut = copy.copy(yIn)

  """If the TTR field exists, then the schemeData structure should have been initialized."""
  if(isfield(schemeDataOut, 'ttr')):
    assert isfield(schemeDataOut, 'ttrLastY')
    assert isfield(schemeDataOut, 'ttrLastT')

    "Find the nodes whose signs have changed."
    changed = np.logical_and(np.nonzero(yIn <= 0)[0], np.nonzero(schemeDataOut.ttrLastY > 0)[0])

    "For those nodes, linearly interpolate the crossing time."
    schemeDataOut.ttr[changed] = schemeDataOut.ttrLastT - \
            np.divide((t - schemeDataOut.ttrLastT) @ schemeDataOut.ttrLastY[changed],\
                        (yIn(changed) - schemeDataOut.ttrLastY(changed)))

    "Record the auxiliary data for next timestep."
    schemeDataOut.ttrLastY = copy.copy(yOut)
    schemeDataOut.ttrLastT = copy.copy(t)
    
  else: 
    "If the TTR field is missing, then we need to initialize the TTR function and the auxiliary fields."
    "Set TTR to be current time if the node is inside the zero sublevel set, +inf otherwise."
    schemeDataOut.ttr = inf * np.ones_like(yIn)
    schemeDataOut.ttr[np.nonzero(yIn <= 0)] = t

    "Record the auxiliary functions."
    schemeDataOut.ttrLastY = yOut
    schemeDataOut.ttrLastT = t

    return yOut, schemeDataOut