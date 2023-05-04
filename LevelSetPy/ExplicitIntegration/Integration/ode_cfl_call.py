__all__ = ["odeCFLcallPostTimestep"]

__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__credits__  	= "There are None."
__license__ 	= "MIT License"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"


import copy
from LevelSetPy.Utilities import *

def odeCFLcallPostTimestep(t, yIn, schemeDataIn, options):
    """
     odeCFLcallPostTimestep: call any postTimestep routines.

     [ yOut, schemeDataOut ] = ...
                            odeCFLcallPostTimestep(t, yIn, schemeDataIn, options)


     Calls one or more postTimestep routines, depending on the contents
       of the options.postTimestep field.  Helper routine for odeCFLn.

     If options.postTimestep is a cell vector of function handles, the
       function are called in order.

     parameters:
       t              Current time.
       yIn            Inp.t version of the level set function, in vector form.
       schemeDataIn   Inp.t version of a structure.
       options        An option structure generated by odeCFLset
                        (use [] as a placeholder if necessary).

       yOut           Output version of the level set function, in vector form.
       schemeDataOut  Output version of the structure.

     The postTimestep routines called will determine whether and how
       yOut and schemeDataOut differ from their input versions.

     Lekan 08/21/2021
    """

    # Copy over the current version of data and scheme structure.
    yOut = copy.copy(yIn)
    schemeDataOut = copy.copy(schemeDataIn)

    # Check to see if there is anything to do.
    if not (options):
        return yOut, schemeDataOut

    if isfield(options, 'postTimestep') and options.postTimestep:
        return yOut, schemeDataOut

    # Make the necessary calls.
    if callable(options.postTimestep):
        yOut, schemeDataOut = options.postTimestep(t, yOut, schemeDataOut)
    elif(isinstance(options.postTimestep, list)):
        for i in range(len(options.postTimestep)):
           yOut, schemeDataOut = options.postTimestep[i](t, yOut, schemeDataOut)

    return yOut, schemeDataOut
