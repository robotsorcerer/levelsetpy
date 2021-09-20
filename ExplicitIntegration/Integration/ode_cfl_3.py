import sys
from .ode_cfl_set import odeCFLset
from .ode_cfl_call import odeCFLcallPostTimestep
from Utilities import *

def odeCFL3(schemeFunc, tspan, y0, options, schemeData):
    """
     odeCFL3: integrate a CFL constrained ODE (eg a PDE by method of lines).

     [ t, y, schemeData ] = odeCFL3(schemeFunc, tspan, y0, options, schemeData)

     Integrates a system forward in time by CFL constrained timesteps
       using a third order Total Variation Diminishing (TVD) Runge-Kutta
       (RK) scheme.  Details can be found in O&F chapter 3.

     parameters:
       schemeFunc	 Function handle to a CFL constrained ODE system e.g. termLaxFriedrich
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

     The output version of schemeData will normally be identical to the input
       version, and therefore can be ignored.  However, if a PostTimestep
       routine is used (see odeCFLset) then schemeData may be modified during
       integration, and the version of schemeData at tf is returned in this
       output argument.

     Copyright 2005 Ian M. Mitchell (mitchell@cs.ubc.ca).
     This software is used, copied and distributed under the licensing
       agreement contained in the file LICENSE in the top directory of
       the distribution.

     Ian Mitchell, 5/14/03.
     Calling parameters modified to more closely match Matlab's ODE suite
       Ian Mitchell, 2/14/04.
     Modified to allow vector level sets.  Ian Mitchell, 12/13/04.
     Modified to add terminalEvent option, Ian Mitchell, 1/30/05.

     Lekan 08/21/2021
    """
    #---------------------------------------------------------------------------
    # How close (relative) do we need to be to the final time?
    small = 100 * sys.float_info.epsilon

    #---------------------------------------------------------------------------
    # Make sure we have the default options settings
    if not options:
        options = odeCFLset
    #---------------------------------------------------------------------------
    # This routine includes multiple substeps, and the CFL restricted timestep
    #   size is chosen on the first substep.  Subsequent substeps may violate
    #   CFL slightly how much should be allowed before generating a warning?

    # This choice allows 20% more than the user specified CFL number,
    #   capped at a CFL number of unity.  The latter cap may cause
    #   problems if the user is using a very aggressive CFL number.
    safetyFactorCFL = min(1.0, 1.2 * options.factorCFL)

    #---------------------------------------------------------------------------
    # Number of timesteps to be returned.
    numT = len(tspan)

    #---------------------------------------------------------------------------
    # If we were asked to integrate forward to a final time.
    if(numT == 2):
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Is this a vector level set integration?
        if(isinstance(y0, list)):
            numY = len(y0)

            # We need a cell vector form of schemeFunc.
            if(isinstance(schemeFunc, list)):
                schemeFuncCell = schemeFunc
            else:
                schemeFuncCell = [schemeFunc for i in range(numY)]
        else:
            # Set numY, but be careful: ((numY == 1) & iscell(y0)) is possible.
            numY = 1

            # We need a cell vector form of schemeFunc.
            schemeFuncCell = [schemeFunc]
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        t = tspan[0]
        steps = 0
        startTime = cputime()
        stepBound = zeros(numY, 1, dtype=np.float64)
        ydot = [np.nan for i in range(numY)] #ell(numY, 1)
        y = y0

        # print('GOT HERE: ', tspan, len(tspan))
        while(tspan[1] - t >= small * np.abs(tspan[1])):
            # print(f'tspan[1]-t: {tspan[1] - t} small * np.abs(tspan[1]): {small * np.abs(tspan[1])}')
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # First substep: Forward Euler from t_n to t_{n+1}.

            # Approximate the derivative and CFL restriction.
            for i in range(numY):
                # print(i, 'yshap: ', y.shape)
                ydot[i], stepBound[i], schemeData = schemeFuncCell[i](t, y, schemeData)

                # print('stepBound[i]: ', stepBound[i])
                # print('ydot: ', ydot[i].shape)
                # If this is a vector level set, rotate the lists of vector arguments.
                if(iscell(y)):
                    # print('y in cell shp: ', y.shape)
                    y = y[ 1:]

                if(iscell(schemeData)):
                    # print('schemeData: ', len(schemeData))
                    schemeData = schemeData[1:]
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Determine CFL bound on timestep, but not beyond the final time.
            #   For vector level sets, use the most restrictive stepBound.
            #   We'll use this fixed timestep for both substeps..
            # print('stepBound ', stepBound, ' options.maxStep ', options.maxStep)
            deltaT = min(np.min(options.factorCFL*stepBound),  tspan[1] - t, options.maxStep)

            # Take the first substep.
            t1 = t + deltaT
            # print('y first substep: ', y.shape)
            if(iscell(y)):
                y1 = [np.nan for _ in range(numY)] #cell(numY, 1)
                for i in range(numY):
                    y1[i] +=(deltaT * ydot[i])
            else:
                # print('y b4 y1: ', y.shape, ': ydot[0] ', ydot, ydot[0].shape)
                y1 = y + deltaT * ydot[0]
            # print('y1 first substep: ', y1.shape)


            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Second substep: Forward Euler from t_{n+1} to t_{n+2}.

            # Approximate the derivative.
            #   We will also check the CFL condition for gross violation.
            for i in range(numY):
                ydot[i], stepBound[i], schemeData = schemeFuncCell[i](t1, y1, schemeData)

                # If this is a vector level set, rotate the lists of vector arguments.
                if(iscell(y1)):
                    y1 = y1[ 1:, 0 ]

                if(iscell(schemeData)):
                    schemeData = schemeData[ 1:, 0 ]

            # Check CFL bound on timestep:
            #   If the timestep chosen on the first substep violates
            #   the CFL condition by a significant amount, throw a warning.
            #   For vector level sets, use the most restrictive stepBound.
            # Occasional failure should not cause too many problems.
            if(deltaT > min(safetyFactorCFL * stepBound)):
                violation = deltaT / stepBound
                warn(f'Second substep violated CFL effective number {violation}')

            # Take the second substep.
            t2 = t1 + deltaT
            if(iscell(y1)):
                y2 = cell(numY, 1)
                for i in range(numY):
                    y2[i] = y1[i] + deltaT * ydot[i]
            else:
                y2 = y1 + deltaT * ydot[0]

            # Combine t_n and t_{n+2} to get approximation at t_{n+1/2}
            tHalf = 0.25 * (3 * t + t2)
            if(iscell(y2)):
                yHalf = cell(numY, 1)
                for i in range(numY):
                    yHalf[i] = 0.25 * (3 * y[i] + y2[i])
            else:
                yHalf = 0.25 * (3 * y + y2)

            #Third substep: Forward Euler from t_{n+1/2} to t_{n+3/2}.

            # Approximate the derivative.
            #   We will also check the CFL condition for gross violation.
            for i in range(numY):
                ydot[i], stepBound[i], schemeData = schemeFuncCell[i](tHalf, yHalf, schemeData)

            # If this is a vector level set, rotate the lists of vector arguments.
            if(iscell(yHalf)):
                yHalf = yHalf[1:,0]

            if(iscell(schemeData)):
                schemeData = schemeData[ 1:, 0 ]

            # Check CFL bound on timestep:
            #   If the timestep chosen on the first substep violates
            #   the CFL condition by a significant amount, throw a warning.
            #   For vector level sets, use the most restrictive stepBound.
            # Occasional failure should not cause too many problems.
            if(deltaT > min(safetyFactorCFL * stepBound)):
                violation = deltaT / stepBound
                warn(f'Third substep violated CFL effective number {violation}')

            # Take the third substep.
            tThreeHalf = tHalf + deltaT
            if(iscell(yHalf)):
                yThreeHalf = cell(numY, 1)
                for i in range(numY):
                    yThreeHalf[i] = yHalf[i] + deltaT * ydot[i]
            else:
                yThreeHalf = yHalf + deltaT * ydot[0]

            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # If there is a terminal event function registered, we need
            #   to maintain the info from the last timestep.
            if(not(options.terminalEvent)):
                yOld , tOld = y,  t

            # Combine t_n and t_{n+3/2} to get third order approximation of t_{n+1}.
            t = (1/3) * (t + 2 * tThreeHalf)
            if(iscell(yThreeHalf)):
                for i in range(numY):
                    y[i] = (1/3) * (y[i] + 2 * yThreeHalf[i])
            else:
                y = (1/3) * (y + 2 * yThreeHalf)

            steps += 1

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
            info(f'\t{steps} steps in {endTime-startTime} seconds from  {tspan[0]}, to {t}')

    elif(numT > 2):
        # If we were asked for the solution at multiple timesteps.
        t, y, schemeData = odeCFLmultipleSteps(schemeFunc, tspan, y0, options, schemeData)

    #---------------------------------------------------------------------------
    else:
        # Malformed time span.
        error('tspan must contain at least two entries')

    return t, y, schemeData
