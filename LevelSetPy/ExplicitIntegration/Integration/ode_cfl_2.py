__all__ = ["odeCFL2"]

__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__credits__  	= "There are None."
__license__ 	= "Molux Licence"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"


import copy
import time
import numpy as np

from LevelSetPy.Utilities import *
from .ode_cfl_set import odeCFLset
from .ode_cfl_mult import odeCFLmultipleSteps
from .ode_cfl_call import odeCFLcallPostTimestep

def  odeCFL2(schemeFunc, tspan, y0, options=None, schemeData=None):
    """
     odeCFL2: integrate a CFL constrained ODE (eg a PDE by method of lines).
     CFL: [Courant–Friedrichs–Lewy condition](https://en.wikipedia.org/wiki/Courant%E2%80%93Friedrichs%E2%80%93Lewy_condition)

     [ t, y, schemeData ] = odeCFL2(schemeFunc, tspan, y0, options, schemeData)

     Integrates a system forward in time by CFL constrained timesteps
       using a second order Total Variation Diminishing (TVD) Runge-Kutta
       (RK) scheme.  Details can be found in O&F chapter 3.

     parameters:
       schemeFunc	 Function handle to a CFL constrained ODE system
                      (typically an approximation to an HJ term e.g. termRestrictUpdate, see below).
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

            [ ydot, stepBound ] = schemeFunc(t, y, schemeData)

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

     Note that using this routine for integrating HJ PDEs will usually
       require that the data array be turned into a vector before the call
       and reshaped into an array after the call.  Option (2) for tspan should
       not be used in this case because of the excessive memory requirements
       for storing solutions at multiple timesteps.

     The output version of schemeData will normally be identical to the inp.t
       version, and therefore can be ignored.  However, if a PostTimestep
       routine is used (see odeCFLset) then schemeData may be modified during
       integration, and the version of schemeData at tf is returned in this
       output argument.

    Lekan 08/16/2021
    """

    "How close (relative) do we need to be to the final time?"
    small = 100 * eps

    if not options: options = odeCFLset()
    """
        This routine includes multiple substeps, and the CFL restricted timestep
        size is chosen on the first substep.  Subsequent substeps may violate
        CFL slightly how much should be allowed before generating a warning?

        This choice allows 20% more than the user specified CFL number,
        capped at a CFL number of unity.  The latter cap may cause
        problems if the user is using a very aggressive CFL number.

        safetyFactorCFL = min(1.0, 1.2 * options.factorCFL)

        Number of timesteps to be returned.
    """
    numT = len(tspan)

    " If we were asked to integrate forward to a final time."
    if(numT == 2):
        if(iscell(y0)):
            numY = len(y0)

            if(iscell(schemeFunc)):
                schemeFuncCell = schemeFunc
            else:
                schemeFuncCell = [schemeFunc for i in range(numY)]
        else:
            numY = 1

            schemeFuncCell = [schemeFunc]

        t         = tspan[0]
        steps     = 0
        startTime = cputime()
        stepBound = np.zeros((numY), dtype=np.float64)
        ydot      = cell(numY)
        y         = copy.copy(y0)

        while(tspan[1] - t >= small * np.abs(tspan[1])):
            " First substep: Forward Euler from t_n to t_{n+1}. Approximate the derivative and CFL restriction."
            for i in range(numY):
                ydot[i], stepBound[i], schemeData = schemeFuncCell[i](t, y, schemeData)

                # If this is a vector level set, rotate the lists of vector arguments.
                if(iscell(y)): y = y[1:]

                if(iscell(schemeData)): schemeData = schemeData[1:]

            """
             Determine CFL bound on timestep, but not beyond the final time.
               For vector level sets, use the most restrictive stepBound. We'll use this fixed timestep for both substeps.
            """

            deltaT = np.min(np.hstack((options.factorCFL*stepBound,  \
                                        tspan[1] - t, options.maxStep)))

            # Take the first substep.
            t1 = t + deltaT
            if(iscell(y)):
                y1 = cell(numY)
                for i in range(numY):
                    y1[i] +=(deltaT * ydot[i])
            else:
                y1 = y + deltaT * ydot[0]

            """
                Second substep: Forward Euler from t_{n+1} to t_{n+2}.

                Approximate the derivative.
                  We will also check the CFL condition for gross violation.
            """
            for i in range(numY):
                ydot[i], stepBound[i], schemeData = schemeFuncCell[i](t1, y1, schemeData)

                # Rotate the lists of vector arguments.
                if(iscell(y1)):  y1 = y1[1:]

                if(iscell(schemeData)): schemeData = schemeData[1:]

            """
              Check CFL bound on timestep: If the timestep chosen on the first substep violates
              the CFL condition by a significant amount, throw a warning. For vector level sets,
              use the most restrictive stepBound. Occasional failure should not cause too many problems.
            """
            if(deltaT > np.min(options.factorCFL * stepBound)):
                violation = deltaT / np.asarray(stepBound)
                warn(f'Second substep violated CFL effective number {violation}')

            "Take the second substep."
            t2 = t1 + deltaT
            if(iscell(y1)):
                y2 = cell(numY, 1)
                for i in range(numY):
                    y2[i] = y1[i] + deltaT * ydot[i]
            else:
                y2 = y1 + deltaT * ydot[0]

            "If there is a terminal event function registered, we need to maintain the info from the last timestep."
            if (isfield(options, "terminalEvent") and np.logical_not(options.terminalEvent)):
                yOld , tOld = y,  t

            "Average t_n and t_{n+2} to get second order approximation of t_{n+1}."
            t = 0.5 * (t + t2)
            if(iscell(y2)):
                for i in range(numY):
                    y[i] = 0.5 * (y[i] + y2[i])
            else:
                y = 0.5 * (y + y2)

            steps += 1

            "If there is one or more post-timestep routines, call them."
            if isfield(options, 'postTimeStep') and options.postTimeStep:
                y, schemeData = odeCFLcallPostTimestep(t, y, schemeData, options)

            "If we are in single step mode, then do not repeat."
            if(strcmp(options.singleStep, 'on')):
                break

            "If there is a terminal event function, establish initial sign of terminal event vector."
            if (isfield(options, 'terminalEvent') and options.terminalEvent):
                eventValue, schemeData = options.terminalEvent(t, y, tOld, yOld, schemeData)

                if((steps > 1) and np.any(np.sign(eventValue) != np.sign(eventValueOld))):
                    break
                else:
                    eventValueOld = eventValue

        endTime = cputime()

        if(strcmp(options.stats, 'on')):
            info(f'{steps} steps in {(endTime-startTime):.2} seconds from  {tspan[0]:.2f} to {t:.2f}.')
    #---------------------------------------------------------------------------
    elif(numT > 2):
        t, y, schemeData = odeCFLmultipleSteps(odeCFL2, schemeFunc, tspan, y0, options, schemeData)

    #---------------------------------------------------------------------------
    else:
        ValueError('tspan must contain at least two entries')

    return t, y, schemeData
