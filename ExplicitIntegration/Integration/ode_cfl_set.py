__all__ = ["odeCFLset"]

from LevelSetPy.Utilities import *

def odeCFLset(kwargs=None):
    """
     odeCFLset: Create/alter options for CFL constrained ode integrators.
     CFL: Courant–Friedrichs–Lewy condition

     See: https://en.wikipedia.org/wiki/Courant%E2%80%93Friedrichs%E2%80%93Lewy_condition
      Inp.ts:
        options = {'name1': value1, 'name2': value2, ...}
      Output:
        options = odeCFLset('name1', value1, 'name2', value2, ...)
        options = odeCFLset(oldopts, 'name1', value1, ...)

     Creates a new options structure (or alters an old one) for CFL
       constrained ODE integrators.  Basically the same as Matlab's odeset
       but with not nearly as many options.

     If called with no inp.t or output parameters, then all options,
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

       PostTimeStep  Function handle to a routine with prototype
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
                       The inp.t parameters include the state and time from
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
                       called after all postTimeStep functions.
                     Defaults to [], which calls no function.

     Lekan: August 16, 2021
    """
    #---------------------------------------------------------------------------
    # No output, no inp.t means caller just wants a list of available options.
    if not kwargs:
        info(f'    factorCFL: [ positive scalar {0.5} ]')
        info('      maxStep: [ positive scalar {REALMAX} ]')
        info(f' postTimeStep: [ function handle | '
                'cell vector of function handles | {[]} ]')
        info('   singleStep: [ on | {off} ]')
        info('        stats: [ on | {off} ]')
        info('terminalEvent: [ function handle | {[]} ]')
        raise ValueError(f'kwargs cannot be {None}')
        return
    else:
        assert isbundle(kwargs), "kwargs must be a bundle type."

    options = Bundle({})
    options.factorCFL = kwargs.__dict__.get('factorCFL', 0.5)
    options.maxStep  = kwargs.__dict__.get('realmax', realmax)
    options.postTimeStep = kwargs.__dict__.get('postTimeStep', None)
    options.singleStep = kwargs.__dict__.get("singleStep", 'off')
    options.stats = kwargs.__dict__.get("stats", 'off')
    options.terminalEvent = kwargs.__dict__.get("terminalEvent", None)
    startArg = 0

    #---------------------------------------------------------------------------
    # Loop through remaining name value pairs
    if options.factorCFL < 0.0:
        raise ValueError('FactorCFL must be a positive scalar double value')
    if options.maxStep < 0.0:
        raise ValueError('MaxStep must be a positive scalar double value')

    if options.postTimeStep is not None:
        if (isinstance(options.postTimeStep, list)):
            for j in range(len(options.postTimeStep)):
                if(not hasattr(options.postTimeStep[j], '__call__')):
                    raise ValueError('Each element in a postTimeStep cell vector must '
                    'be a function handle.')
        else:
            raise ValueError('PostTimeStep parameter must be a function handle or '
                                'a list of function handles.')

    # print('options.singleStep ', options.singleStep)
    if (not isinstance(options.singleStep, str) and \
            (not strcmp(options.singleStep, 'on') or \
            not strcmp(options.singleStep, 'off'))):
        raise ValueError('SingleStep must be one of the strings ''on'' or ''off''')

    if(not isinstance(options.stats, str) and not \
        strcmp(options.stats, 'on') or not (options.stats, 'off')):
        raise ValueError('Stats must be one of the strings \'on\' or \'off\'')

    if(options.terminalEvent is not None and not hasattr(options.terminalEvent, '__call__')):
        raise ValueError('PostTimeStep parameter must be a function handle.')

    return options
