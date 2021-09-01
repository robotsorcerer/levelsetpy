from utils import *
from .ode_cfl_set import odeCFLset

def odeCFLget(options=None, name=None):
    """
     odeCFLget: Get option values for CFL constrained ode integrators.

       value = odeCFLget(options, 'name')

     Retrieves the value of an option parameter from a CFL constrained
       ODE integrator option structure.  Note that the type of the returned
       value depends on the named option.

     If called with no input or output parameters, then all options,
       their valid values and defaults are listed.

     Parameter options may be the empty vector [], in which case the default
       value of the named option is returned.

     Available options (options names are case insensitive):

       FactorCFL    Scalar by which to multiply CFL timestep bound in order
                      to determine the timestep to actually take.
                      Typically in range (0,1), default = 0.5
                      choose 0.9 for aggressive integration.

       MaxStep      Maximum step size (independent of CFL).
                      Default is REALMAX.

       PostTimestep Function handle to a routine with prototype
                           [ yOut schemeDataOut ] = f(t, yIn, schemeDataIn)
                      which is called after every timestep and can be used
                      to modify the state vector y or to modify or record
                      information in the schemeData structure.
                    May also be a cell vector of such function handles, in
                      which case all function handles are called in order
                      after each timestep.
                    Defaults to [], which calls no function.

       SingleStep   Specifies whether to exit integrator after a single
                      CFL constrained timestep (for debugging).
                      Either 'on' or 'off', default = 'off'.

       Stats        Specifies whether to display statistics.
                      Either 'on' or 'off', default = 'off'.

       TerminalEvent Function handle to a routine with prototype
                            [ value, schemeDataOut ] = ...
                                         f(t, y, deltaT, updateY, schemeDataIn)
                       which is called after every timestep and can be used to
                       halt time integration before the final time is reached.
                       The input parameters include the state and time from
                       the previous timestep.  If any element of the
                       return parameter value changes sign from one timestep
                       to the next, then integration is terminated and
                       control is returned to the calling function.
                     Integration cannot be terminated in this manner until
                       after at least two timesteps.
                     Unlike Matlab's ODE event system, no attempt is made
                       to accurately locate the time at which the event
                       function passed through zero.
                     If both are present, the terminalEvent function will be
                       called after all postTimestep functions.
                     Defaults to [], which calls no function.

     Copyright 2005 Ian M. Mitchell (mitchell@cs.ubc.ca).
     This software is used, copied and distributed under the licensing
       agreement contained in the file LICENSE in the top directory of
       the distribution.

     Ian Mitchell, 8/30/04
     Modified to add terminalEvent option, Ian Mitchell, 1/30/05

     Lekan 08/21/21
    """
    #No output, no input means caller just wants a list of available options.
    if not options and not name:
        return odeCFLset


    # If first input is an empty vector, get the default option structure.
    if not options:
        options = odeCFLset

    # Return the appropriate option value.

    # Remember that the case labels are lower case.
    if name.lower() =='factorcfl':
        value = options.factorCFL
    elif name.lower() =='maxstep':
        value = options.maxStep
    elif name.lower() =='posttimestep':
        value = options.postTimestep
    elif name.lower() =='singlestep':
        value = options.singleStep
    elif name.lower() =='stats':
        value = options.stats
    elif name.lower() =='terminalevent':
        value = options.terminalEvent
    else:
        error('Unknown odeCFL option %s', name)

    return value
