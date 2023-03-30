__all__ = ["odeCFLcallPostTimestep"]

import copy
import numpy as np
from LevelSetPy.Utilities import *


def odeCFLcallPostTimestep(t, yIn, schemeDataIn, options):
  """
   odeCFLcallPostTimestep: call any postTimestep routines.
  
   [ yOut, schemeDataOut ] = ...
                          odeCFLcallPostTimestep(t, yIn, schemeDataIn, options);
  
  
   Calls one or more postTimestep routines, depending on the contents
     of the options.postTimestep field.  Helper routine for odeCFLn.
  
   If options.postTimestep is a cell vector of function handles, the
     function are called in order.
  
   parameters:
     t              Current time.
     yIn            Input version of the level set function, in vector form.
     schemeDataIn   Input version of a structure.
     options        An option structure generated by odeCFLset 
                      (use [] as a placeholder if necessary).
  
     yOut           Output version of the level set function, in vector form.
     schemeDataOut  Output version of the structure.
  
   The postTimestep routines called will determine whether and how
     yOut and schemeDataOut differ from their input versions.
  """
  # Copy over the current version of data and scheme structure.
  yOut = copy.copy(yIn)
  schemeDataOut = copy.copy(schemeDataIn)

  # Check to see if there is anything to do.
  if(options is None or not isfield(options, "postTimestep")):
    return None

  # Make the necessary calls.
  if(isfield(options, 'postTimestep')):
    yOut, schemeDataOut  = options.postTimestep(t, yOut, schemeDataOut)
  elif(iscell(options.postTimestep)):
    for i in range(len(options.postTimestep)):
      yOut, schemeDataOut = options.postTimestep[i](t, yOut, schemeDataOut)

  return yOut, schemeDataOut
                        