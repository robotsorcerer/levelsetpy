import sys, os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import numpy as np
from math import pi
from utils import expand, zeros, Bundle, ones
from Grids import createGrid
from ValueFuncs import proj, HJIPDE_solve, eval_u
from Visualization import visSetIm
from InitialConditions import shapeCylinder
from DynamicalSystems import *
import matplotlib.pyplot as plt


import logging
logger = logging.getLogger(__name__)

def main():
    """
      Reproduces Sylvia's Tutorial on BRS
         1. Run Backward Reachable Set (BRS) with a goal
             uMode = 'min' <-- goal
             minWith = 'none' <-- Set (not tube)
             compTraj = false <-- no trajectory
         2. Run BRS with goal, then optimal trajectory
             uMode = 'min' <-- goal
             minWith = 'none' <-- Set (not tube)
             compTraj = True <-- compute optimal trajectory
         3. Run Backward Reachable Tube (BRT) with a goal, then optimal trajectory
             uMode = 'min' <-- goal
             minWith = 'minVOverTime' <-- Tube (not set)
             compTraj = True <-- compute optimal trajectory
         4. Add disturbance
             dStep1: define a dMax (dMax = [.25, .25, 0];)
             dStep2: define a dMode (opposite of uMode)
             dStep3: input dMax when creating your DubinsCar
             dStep4: add dMode to schemeData
         5. Change to an avoid BRT rather than a goal BRT
             uMode = 'max' <-- avoid
             dMode = 'min' <-- opposite of uMode
             minWith = 'minVOverTime' <-- Tube (not set)
             compTraj = false <-- no trajectory
         6. Change to a Forward Reachable Tube (FRT)
             add schemeData.tMode = 'forward'
             note: now having uMode = 'max' essentially says "see how far I can
             reach"
         7. Add obstacles
             add the following code:
             obstacles = shapeCylinder(g, 3, [-1.5; 1.5; 0], 0.75);
             HJIextraArgs.obstacles = obstacles;
         8. Add random disturbance (white noise)
             add the following code:
             HJIextraArgs.addGaussianNoiseStandardDeviation = [0; 0; 0.5];
    """

    ## Should we compute the trajectory?
    compTraj = True;

    ## Grid
    grid_min = expand(np.array((-5, -5, -pi)), ax = 1); # Lower corner of computation domain
    grid_max = expand(np.array((5, 5, -pi)), ax = 1);   # Upper corner of computation domain
    N = 41*ones(3, 1) #expand(np.array((41, 41,  41)), ax = 1);        # Number of grid points per dimension
    pdDims = 3;               # 3rd dimension is periodic
    g = createGrid(grid_min, grid_max, N, pdDims);
    # Use "g = createGrid(grid_min, grid_max, N);" if there are no periodic
    # state space dimensions

    ## target set
    R = 1;
    # data0 = shapeCylinder(grid,ignoreDims,center,radius)
    data0 = shapeCylinder(g, 3, zeros(3, 1), R);
    # also try shapeRectangleByCorners, shapeSphere, etc.

    ## time vector
    t0 = 0;
    tMax = 2;
    dt = 0.05;
    tau = np.arange(t0, tMax+dt, dt); # account for pythonb's 0-indexing

    ## problem parameters

    # input bounds
    speed = 1;
    wMax = 1;
    # do dStep1 here

    # control trying to min or max value function?
    uMode = 'min';
    # do dStep2 here

    ## Pack problem parameters

    #do dStep3 here
    new_params = dict(x=zeros(1, 3), wRange=[-wMax, wMax], speed=speed, \
                      xhist=zeros(1, 3), uhist=zeros(1,3))
    dubins_default_params.update(new_params)

    # Define dynamic system
    dCar = DubinsCar(**dubins_default_params)

    # Put grid and dynamic systems into schemeData
    schemeData = Bundle(dict(grid = g, dynSys = dCar, accuracy = 'high',
                            uMode = uMode))
    #do dStep4 here
    print(schemeData)

    ## additive random noise
    #do Step8 here
    #HJIextraArgs.addGaussianNoiseStandardDeviation = [0; 0; 0.5];
    # Try other noise coefficients, like:
    #    [0.2; 0; 0]; # Noise on X state
    #    [0.2,0,0;0,0.2,0;0,0,0.5]; # Independent noise on all states
    #    [0.2;0.2;0.5]; # Coupled noise on all states
    #    {zeros(size(g.xs{1})); zeros(size(g.xs{1})); (g.xs{1}+g.xs{2})/20}; # State-dependent noise

    ## If you have obstacles, compute them here

    ## Compute value function
    HJIextraArgs = Bundle(dict(
                            visualize = Bundle(dict(
                            valueSet = 1,
                            initialValueSet = 1,
                            figNum = 1, #set figure number
                            deleteLastPlot = True, #delete previous plot as you update
                            plotData = Bundle(dict(
                            # comment these if you don't want to see a 2D slice
                            plotDims = [1, 1, 0], #plot x, y
                            projpt = [0], #project at theta = 0
                            )),
                            viewAngle = [0,90], # view 2D
                            )),
                            ))

    #[data, tau, extraOuts] = ...
    # HJIPDE_solve(data0, tau, schemeData, minWith, extraArgs)
    [data, tau2, _] = HJIPDE_solve(data0, tau, schemeData, 'None', HJIextraArgs);

    ## Compute optimal trajectory from some initial state
    if compTraj:

        #set the initial state
        xinit = np.array(([[2, 2, -pi]]))

        #check if this initial state is in the BRS/BRT
        #value = eval_u(g, data, x)
        value = eval_u(g,data[:,:,:,-1],xinit);

        if value <= 0: #if initial state is in BRS/BRT
            # find optimal trajectory

            dCar.x = xinit; #set initial state of the dubins car

            TrajextraArgs = Bundle(dict(
                                uMode = uMode, #set if control wants to min or max
                                dMode = 'max',
                                visualize = True, #show plot
                                fig_num = 2, #figure number
                                #we want to see the first two dimensions (x and y)
                                projDim = np.array([[1, 1, 0]])
                            ))


            #flip data time points so we start from the beginning of time
            dataTraj = np.flip(data,4); # I hope this is correct

            # [traj, traj_tau] = ...
            # computeOptTraj(g, data, tau, dynSys, extraArgs)
            [traj, traj_tau] = computeOptTraj(g, dataTraj, tau2, dCar, TrajextraArgs);

            # fig = plt.gcf()
            # plt.clf()
            # h = visSetIm(g, data[:,:,:,-1]);
            # # h.FaceAlpha = .3;
            #
            # ax = fig.add_subplot(projection='3d')
            # ax.scatter(xinit[1], xinit[2], xinit[3]);
            # # s.SizeData = 70;
            # ax.set_title('The reachable set at the end and x_init')
            #
            # #plot traj
            # # figure(4)
            # ax2 = fig.add_subplot(1, 1, 1)
            # ax2.plot(traj[0,:], traj[1,:])
            # ax2.set_xlim(left=-5, right=5)
            # ax2.set_ylim(left=-5, right=5)
            # add the target set to that
            g2D, data2D = proj(g, data0, [0, 0, 1]);
            # visSetIm(g2D, data2D, 'green');
            # ax2.set_title('2D projection of the trajectory & target set')
            # hold off
        else:
            logger.fatal(f'Initial state is not in the BRS/BRT! It have a value of {value}')
        return g2D, data2D


if __name__ == '__main__':
    main()
