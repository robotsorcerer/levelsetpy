import sys
from Utilities import *

def odeCFLset(kwargs):
    """
     odeCFLset: Create/alter options for CFL constrained ode integrators.
     CFL: Courant–Friedrichs–Lewy condition

     See: https://en.wikipedia.org/wiki/Courant%E2%80%93Friedrichs%E2%80%93Lewy_condition
      Inputs:
        kwargs = {'name1': value1, 'name2': value2, ...}
      Output:
        options = odeCFLset('name1', value1, 'name2', value2, ...)
        options = odeCFLset(oldopts, 'name1', value1, ...)

     Creates a new options structure (or alters an old one) for CFL
       constrained ODE integrators.  Basically the same as Matlab's odeset
       but with not nearly as many options.

     If called with no input or output parameters, then all options,
       their valid values and defaults are listed.

     Available options (options names are case insensitive):

       FactorCFL     Scalar by which to multiply CFL timestep bound in order
                       to determine the timestep to actually take.
                       Typically in range (0,1), default = 0.5
                       choose 0.9 for aggressive integration. This is alpha in
                       (3.8-3.10), Osher and Fedkiw. It helps establish the stability condition
                       such that small errors in the approx are not amplified as the solution
                       is marched forward in time.

       MaxStep       Maximum step size (independent of CFL).
                       Default is sys.float_info.max.

       PostTimestep  Function handle to a routine with prototype
                            [ yOut, schemeDataOut ] = f(t, yIn, schemeDataIn)
                       which is called after every timestep and can be used
                       to modify the state vector y or to modify or record
                       information in the schemeData structure.
                     May also be a cell vector of such function handles, in
                       which case all function handles are called in order
                       after each timestep.
                     Defaults to [], which calls no function.

       SingleStep    Specifies whether to exit integrator after a single
                       CFL constrained timestep (for debugging).
                       Either 'on' or 'off', default = 'off'.

       Stats         Specifies whether to display statistics.
                       Either 'on' or 'off', default = 'off'.

       TerminalEvent Function handle to a routine with prototype
                    [ value, schemeDataOut ] = f(t, y, tOld, yOld, schemeDataIn)
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

     Copyright 2005-2008 Ian M. Mitchell (mitchell@cs.ubc.ca).
     This software is used, copied and distributed under the licensing
       agreement contained in the file LICENSE in the top directory of
       the distribution.

     Lekan: August 16, 2021
    """
    #---------------------------------------------------------------------------
    # No output, no input means caller just wants a list of available options.
    if not kwargs:
        info(f'    factorCFL: [ positive scalar {0.5} ]')
        info('      maxStep: [ positive scalar {REALMAX} ]')
        info(f' postTimestep: [ function handle | '
                'cell vector of function handles | {[]} ]')
        info('   singleStep: [ on | {off} ]')
        info('        stats: [ on | {off} ]')
        info('terminalEvent: [ function handle | {[]} ]')
        return

        #---------------------------------------------------------------------------
    # First input argument is an old options structure
    if len(kwargs) > 0 and 'options' in kwargs.__dict__.keys():
        options = kwargs.options
        startArg = 1
    else:
        # Create the default options structure.
        options = Bundle(dict())
        options.factorCFL = 0.5
        options.maxStep = sys.float_info.max
        options.postTimestep = []
        options.singleStep = 'off'
        options.stats = 'off'
        options.terminalEvent = []
        startArg = 0

    #---------------------------------------------------------------------------
    # Loop through remaining name value pairs
    keys = list(kwargs.__dict__.keys())
    for i in range(startArg, len(kwargs)):
        name = keys[i]
        value = kwargs.__dict__[name]

        # Remember that the case labels are lower case.
        if name.lower()=='factorcfl':
            if(isfloat(value) and (len([value]) == 1) and (value > 0.0)):
                options.factorCFL = value
            else:
                error('FactorCFL must be a positive scalar double value')

        elif name.lower()== 'maxstep':
            if(isfloat(value) and (len([value]) == 1) and (value > 0.0)):
                options.maxStep = value
            else:
                error('MaxStep must be a positive scalar double value')

        elif name.lower()==  'posttimestep':
            if(hasattr(value, '__call__') or value is None):
                options.postTimestep = value
            elif(isinstance(value, 'cell')):
                for j in range(length(value)):
                    if(not hasattr(value[j], '__call__')):
                        error('Each element in a postTimestep cell vector must '
                        'be a function handle.')
                options.postTimestep = value
            else:
                error('PostTimestep parameter must be a function handle or '
                    'a cell vector of function handles.')

        elif name.lower()==  'singlestep':
            if(isinstance(value, str) and (value=='on') or (value=='off')):
                options.singleStep = value
            else:
                error('SingleStep must be one of the strings ''on'' or ''off''')

        elif name.lower()==  'stats':
            if(isinstance(value, str) and (value=='on') or (value=='off')):
                options.stats = value
            else:
                error('Stats must be one of the strings \'on\' or \'off\'')

        elif name.lower()==  'terminalevent':
            if(value is None or hasattr(value, '__call__')):
                options.terminalEvent = value
            else:
                error('PostTimestep parameter must be a function handle.')
        else:
            error(f'Unknown odeCFL option, {name}')

    return options
