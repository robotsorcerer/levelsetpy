__all__ = ["odeCFL1"]

import cupy as cp
import numpy as np
from LevelSetPy.Utilities import *
from .ode_cfl_set import odeCFLset
from .ode_cfl_call import odeCFLcallPostTimestep

def odeCFL1(schemeFunc, tspan, y0, options=None, schemeData=None):
    """
     odeCFL1: integrate a CFL constrained ODE (eg a PDE by method of lines).

     [ t, y, schemeData ] = odeCFL1(schemeFunc, tspan, y0, options, schemeData)

     Integrates a system forward in time by CFL constrained timesteps
       using a first order forward Euler scheme
       (which happens to be the first order TVD RK scheme).

     parameters:
       schemeFunc	 Function handle to a CFL constrained ODE system
                      (typically an approximation to an HJ term, see below).
       tspan        Range of time over which to integrate (see below).
       y0           Initial condition vector
                      (typically the data array in vector form).
       options      An option structure generated by odeCFLset
                      (use [] as a placeholder if necessary).
       schemeData   Structure passed through to schemeFunc.


       t            Output time(s) (see below).
       y            Output state (see below).
       schemeData   Output version of schemeData (see below).

     A CFL constrained ODE system is described by a function with prototype

            [ ydot, stepBound, schemeData ] = schemeFunc(t, y, schemeData)

       where t is the current time, y the current state vector and
       schemeData is passed directly through.  The output stepBound
       is the maximum allowed time step that will be taken by this function
       (typically the option parameter factorCFL will choose a smaller step size).

     The time interval tspan may be given as
       1) A two entry vector [ t0 tf ], in which case the output will
          be scalar t = tf and a row vector y = y(tf).
       2) A vector with three or more entries, in which case the output will
          be column vector t = tspan and each row of y will be the solution
          at one of the times in tspan.  Unlike Matlab's ode suite routines,
          this version just repeatedly calls version (1), so it is not
          particularly efficient.

     Depending on the options specified, the final time may not be reached.
       If integration terminates early, then t (in tspan case (1)) or t(end)
       (in tspan case(2)) will contain the final time reached.

     Note that using this routine for integrating HJ PDEs will usually
       require that the data array be turned into a vector before the call
       and reshaped into an array after the call.  Option (2) for tspan should
       not be used in this case because of the excessive memory requirements
       for storing solutions at multiple timesteps.

     The output version of schemeData will normally be identical to the inp.t
       version, and therefore can be ignored.  However, it is possible for
       schemeFunc or a PostTimestep routine (see odeCFLset) to modify the
       structure during integration, and the version of schemeData at tf is
       returned in this output argument.

     Lekan Molu, 08/21/2021
    """
    small = 100 * eps
    #---------------------------------------------------------------------------
    # Make sure we have the default options settings
    if not options:
        options = odeCFLset()

    # Number of timesteps to be returned.
    numT = len(tspan)
    #---------------------------------------------------------------------------
    # If we were asked to integrate forward to a final time.
    if(numT == 2):
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Is this a vector level set integration?
        if(iscell(y0)):
            numY = len(y0)
            # We need a cell vector form of schemeFunc.
            if(iscell(schemeFunc)):
                schemeFuncCell = schemeFunc
            else:
                schemeFuncCell = [schemeFunc for i in range(numY)]
        else:
            # Set numY, but be careful: ((numY == 1) & iscell(y0)) is possible.
            numY = 1
            # We need a cell vector form of schemeFunc.
            schemeFuncCell = schemeFunc
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        t = tspan[0]
        steps = 0; startTime = cputime(); stepBound = np.zeros((numY), dtype=np.float64)
        ydot = cell(numY, 1); y = copy.copy(y0)

        while(tspan[1] - t >= small * np.abs(tspan[1])):
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # First substep: Forward Euler from t_n to t_{n+1}.

            # Approximate the derivative and CFL restriction.
            for i in range(numY):
                ydot[i], stepBound[i], schemeData = schemeFuncCell[i](t, y, schemeData)
                # If this is a vector level set, rotate the lists of vector arguments.
                if(iscell(y)):
                    y = y[1:]

                if(iscell(schemeData)):
                    schemeData = schemeData[1:]
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Determine CFL bound on timestep, but not beyond the final time.
            #   For vector level sets, use the most restrictive stepBound.
            #   We'll use this fixed timestep for both substeps.
            deltaT = np.min(np.hstack((options.factorCFL*stepBound,  \
                                        tspan[1] - t, options.maxStep)))
            # If there is a terminal event function registered, we need
            #   to maintain the info from the last timestep.
            if options.terminalEvent:
                yOld , tOld = y, t
            # Update time.
            t += deltaT
            # Update level set functions.
            if(iscell(y)):
                for i in range(numY):
                    y1[i] +=(deltaT * ydot[i])
            else:
                y1 = y + deltaT * ydot[0]
            steps += 1
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # If there is one or more post-timestep routines, call them.
            if options.postTimestep:
              y, schemeData = odeCFLcallPostTimestep(t, y, schemeData, options)

            # If we are in single step mode, then do not repeat.
            if(strcmp(options.singleStep, 'on')):
                break

            # If there is a terminal event function, establish initial sign
            #   of terminal event vector.
            if options.terminalEvent:
                eventValue, schemeData = options.terminalEvent(t, y, tOld, yOld, schemeData)

                if((steps > 1) and np.any(np.sign(eventValue) != np.sign(eventValueOld))):
                    break
                else:
                    eventValueOld = eventValue

        endTime = cputime()
        if(strcmp(options.stats, 'on')):
            info(f'{steps} steps in {(endTime-startTime):.2} seconds from  {tspan[0]} to {t}.')
        elif(numT > 2):
            # If we were asked for the solution at multiple timesteps.
            t, y, schemeData = odeCFLmultipleSteps(schemeFunc, tspan, y0, options, schemeData)
        else:
            # Malformed time span.
            error('tspan must contain at least two entries')


    return t, y, schemeData
