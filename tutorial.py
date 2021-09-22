import copy
from absl import flags, app
import sys, os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import numpy as np
from math import pi
from Utilities import expand, zeros, Bundle, ones, error
from Grids import createGrid
from ValueFuncs import *
from Visualization import *
from InitialConditions import shapeCylinder
from DynamicalSystems import *
from SpatialDerivative import upwindFirstENO3a
import matplotlib as mpl
mpl.use("Qt5Agg")
import matplotlib.pyplot as plt

flags.DEFINE_boolean('verbose', False, 'How much debug info to print.')
# General variables now in flags
flags.DEFINE_boolean('matplots', 1, 'Show plots?')
flags.DEFINE_boolean('hj_progress', False, 'Display optimization progress')

FLAGS = flags.FLAGS
FLAGS(sys.argv) # we need to explicitly to tell flags library to parse argv before we can access FLAGS.xxx.

import logging
if FLAGS.verbose:
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
else:
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

logger = logging.getLogger(__name__)



def main(argv):
    """
      Reproduces Sylvia's Tutorial on BRS
         1. Run Backward Reachable Set (BRS) with a goal
             uMode = 'min' <-- goal
             minWith = 'none' <-- Set (not tube)
             compTraj = False <-- no trajectory
         2. Run BRS with goal, then optimal trajectory
             uMode = 'min' <-- goal
             minWith = 'none' <-- Set (not tube)
             compTraj = True <-- compute optimal trajectory
         3. Run Backward Reachable Tube (BRT) with a goal, then optimal trajectory
             uMode = 'min' <-- goal
             minWith = 'minVOverTime' <-- Tube (not set)
             compTraj = True <-- compute optimal trajectory
         4. Add disturbance
             dStep1: define a dMax (dMax = [.25, .25, 0])
             dStep2: define a dMode (opposite of uMode)
             dStep3: input dMax when creating your DubinsCar
             dStep4: add dMode to schemeData
         5. Change to an avoid BRT rather than a goal BRT
             uMode = 'max' <-- avoid
             dMode = 'min' <-- opposite of uMode
             minWith = 'minVOverTime' <-- Tube (not set)
             compTraj = False <-- no trajectory
         6. Change to a Forward Reachable Tube (FRT)
             add schemeData.tMode = 'forward'
             note: now having uMode = 'max' essentially says "see how far I can
             reach"
         7. Add obstacles
             add the following code:
             obstacles = shapeCylinder(g, 3, [-1.5;' 1.5;' 0], 0.75)
             HJIextraArgs.obstacles = obstacles
         8. Add random disturbance (white noise)
             add the following code:
             HJIextraArgs.addGaussianNoiseStandardDeviation = [0 0 0.5]
    """

    ## Should we compute the trajectory?
    compTraj = True

    ## Grid
    grid_min = expand(np.array((-5, -5, -pi)), ax = 1) # Lower corner of computation domain
    grid_max = expand(np.array((5, 5, pi)), ax = 1)   # Upper corner of computation domain
    N = 41*ones(3, 1).astype(int) #expand(np.array((41, 41,  41)), ax = 1)        # Number of grid points per dimension
    pdDims = 2              # 3rd dimension is periodic
    g = createGrid(grid_min, grid_max, N, pdDims)
    # print(f'g.vs after creation: {[x.shape for x in g.vs]}')

    ## target set
    data0 = shapeCylinder(g, 2, zeros(len(N), 1, np.float64), radius=1)
    # show3D(g=g, mesh=data0, savedict={"save":False})
    # also try shapeRectangleByCorners, shapeSphere, etc.
    # print(f'g.vs after cyl: {[x.shape for x in g.vs]}')

    ## time vector
    t0 = 0
    tMax =  2
    dt = 0.05
    tau = np.arange(t0, tMax+dt, dt) # account for pythonb's 0-indexing

    ## problem parameters

    # input bounds
    speed = 1
    wMax = 1
    # do dStep1 here

    # control trying to min or max value function?
    uMode = 'min'
    # do dStep2 here

    ## Pack problem parameters

    # Define dynamic system
    # dCar = DubinsCar(dubins_default_params)
    dCar = DubinsCar(np.zeros((3,1)), wMax, speed)

    # Put grid and dynamic systems into schemeData
    schemeData = Bundle(dict(grid = g, dynSys = dCar, accuracy = 'high',
                            uMode = uMode))
    #do dStep4 here
    # print(f'g.vs after scheme: {[x.shape for x in g.vs]}')

    ## additive random noise
    #do Step8 here
    #HJIextraArgs.addGaussianNoiseStandardDeviation = [0 0 0.5]
    # Try other noise coefficients, like:
    #    [0.2 0 0] # Noise on X state
    #    [0.2,0,00,0.2,00,0,0.5] # Independent noise on all states
    #    [0.20.20.5] # Coupled noise on all states
    #    {zeros(size(g.xs{1})) zeros(size(g.xs{1})) (g.xs{1}+g.xs{2})/20} # State-dependent noise

    ## If you have obstacles, compute them here

    ## Compute value function
    HJIextraArgs = Bundle(dict(
                            # visualize = Bundle(dict(
                            # valueSet = True,
                            # initialValueSet = True,
                            # figNum = 1, #set figure number
                            # deleteLastPlot = True, #delete previous plot as you update
                            # plotData = Bundle(dict(
                            # # comment these if you don't want to see a 2D slice
                            # plotDims = [1, 1, 0], #plot x, y
                            # projpt = [0], #project at theta = 0
                            # )),
                            # viewAngle = [0,90], # view 2D
                            # )),
                            ))

    data, tau2, _ = HJIPDE_solve(data0, tau, schemeData, None, HJIextraArgs)
    # print(f'tau2: {tau2}')
    print(f'data: {data.shape}, data[1:10]: {data[:10, 0, 0, 0]}')

    ## Compute optimal trajectory from some initial state
    if compTraj:

        #set the initial state
        xinit = np.array(([[2, 2, -pi]]))

        #check if this initial state is in the BRS/BRT
        # geval = copy.deepcopy(g)
        value = eval_u(copy.copy(g),data[:,:,:,-1],xinit)
        print(f'value: {value}')
        # print(f'Tut g.vs after eval: {[x.shape for x in g.vs]}')
        # print(f'Tut geval.vs after eval: {[x.shape for x in geval.vs]}')
        if value <= 0: #if initial state is in BRS/BRT
            # find optimal trajectory

            dCar.x = xinit #set initial state of the dubins car

            TrajextraArgs = Bundle(dict(
                                uMode = uMode, #set if control wants to min or max
                                dMode = 'max',
                                visualize = True, #show plot
                                fig_num = 2, #figure number
                                derivFunc= upwindFirstENO3a,
                                #we want to see the first two dimensions (x and y)
                                projDim = np.array([[1, 1, 0]])
                            ))


            #flip data time points so we start from the beginning of time
            dataTraj = np.flip(data,3)

            traj, traj_tau = computeOptTraj(copy.copy(g), dataTraj, tau2, dCar, TrajextraArgs)

            # fig = plt.gcf()
            # plt.clf()
            # h = visSetIm(g, data[:,:,:,-1])
            # # h.FaceAlpha = .3
            #
            # ax = fig.add_subplot(projection='3d')
            # ax.scatter(xinit[1], xinit[2], xinit[3])
            # # s.SizeData = 70
            # ax.set_title('The reachable set at the end and x_init')
            #
            # #plot traj
            # # figure(4)
            # ax2 = fig.add_subplot(1, 1, 1)
            # ax2.plot(traj[0,:], traj[1,:])
            # ax2.set_xlim(left=-5, right=5)
            # ax2.set_ylim(left=-5, right=5)
            # add the target set to that
            g2D, data2D = proj(g, data0, [0, 0, 1])
            # visSetIm(g2D, data2D, 'green')
            # ax2.set_title('2D projection of the trajectory & target set')
            # hold off
        else:
            error(f'Initial state is not in the BRS/BRT! It have a value of {value}')
        return g2D, data2D


if __name__ == '__main__':
    logger.info("Started HJ Optimization")
    app.run(main)
    # main()
