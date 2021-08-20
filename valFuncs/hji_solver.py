import copy
from utils import *
import numpy as np
from datetime import datetime
from hamiltonians import genericHam, genericPartial
from ExplicitIntegration import *
from SpatialDerivative import upwindFirstENO2
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

def HJIPDE_solve(data0, tau, schemeData, compMethod, extraArgs):
    """
     [data, tau, extraOuts] = \
       HJIPDE_solve(data0, tau, schemeData, minWith, extraargs)
         Solves HJIPDE with initial conditions data0, at times tau, and with
         parameters schemeData and extraArgs

     ----- How to use this function -----

     Inputs:
       data0           - initial value function
       tau             - list of computation times
       schemeData      - problem parameters passed into the Hamiltonian func
                           .grid: grid (required!)
                           .accuracy: accuracy of derivatives
       compMethod      - Informs which optimization we're doing
                           - 'set' or 'none' to compute reachable set
                             (not tube)
                           - 'zero' or 'minWithZero' to min Hamiltonian with
                              zero
                           - 'minVOverTime' to do min with previous data
                           - 'maxVOverTime' to do max with previous data
                           - 'minVWithL' or 'minVWithTarget' to do min with
                              targets
                           - 'maxVWithL' or 'maxVWithTarget' to do max with
                              targets
                           - 'minVWithV0' to do min with original data
                             (default)
                           - 'maxVWithV0' to do max with original data
       extraArgs       - this structure can be used to leverage other
                           additional functionalities within this function.
                           Its subfields are:
         .obstacleFunction:    (matrix) a function describing a single
                               obstacle or a list of obstacles with time
                               stamps tau (obstacles must have same time stamp
                               as the solution)
         .targetFunction:      (matrix) the function l(x) that describes a
                               stationary goal/unsafe set or a list of targets
                               with time stamps tau (targets must have same
                               time stamp as the solution). This functionality
                               is mainly useful when the targets are
                               time-varying, in case of variational inequality
                               for example; data0 can be used to specify the
                               target otherwise. This is also useful when
                               warm-starting with a value function (data0)
                               that is not equal to the target/cost function
                               (l(x))
         .keepLast:            (bool) Only keep data from latest time stamp
                               and delete previous datas
         .quiet:               (bool) Don't spit out stuff in command window
         .lowMemory:           (bool) use methods to save on memory
         .fipOutput:           (bool) flip time stamps of output
         .stopInit:            (vector) stop the computation once the
                               reachable set includes the initial state
         .stopSetInclude:      (matrix) stops computation when reachable set
                               includes this set
         .stopSetIntersect:    (matrix) stops computation when reachable set
                               intersects this set
         .stopLevel:           (double) level of the stopSet to check the
                               inclusion for. Default level is zero.
         .stopConverge:        (bool) set to True to stop the computation when
                               it converges
         .convergeThreshold:   Max change in each iteration allowed when
                               checking convergence
         .ignoreBoundary:      Ignores the boundary of the grid when
                               calculating convergence
         .discountFactor:      (double) amount by which you'd like to discount
                               the value function. Used for ensuring
                               convergence. Remember to move your targets
                               function (l(x)) and initial value function
                               (data0) down so they are below 0 everywhere.
                               you can raise above 0 by the same amount at the
                               end.
         .discountMode:        Options are 'Kene' or 'Jaime'.  Defaults to
                               'Jaime'. Math is in Kene's minimum discounted
                               rewards paper and Jaime's "bridging
                               hamilton-jacobi safety analysis and
                               reinforcement learning" paper
         .discountAnneal:      (string) if you want to anneal your discount
                               factor over time so it converges to the right
                               solution, use this.
                                   - 'soft' moves it slowly towards 1 each
                                     time convergence happens
                                   - 'hard' sets it to 1 once convergence
                                      happens
                                   - 1 sets it to 'hard'
         .SDModFunc, .SDModParams:
             Function for modifying scheme data every time step given by tau.
             Currently this is only used to switch between using optimal control at
             every grid point and using maximal control for the SPP project when
             computing FRS using centralized controller

         .saveFilename, .saveFrequency:
             file name under which temporary data is saved at some frequency in
             terms of the number of time steps

         .compRegion:          unused for now (meant to limit computation
                               region)
         .addGaussianNoiseStandardDeviation:
               adds random noise

         .makeVideo:           (bool) whether or not to create a video
         .videoFilename:       (string) filename of video
         .frameRate:           (int) framerate of video
         .visualize:           either fill in struct or set to 1 for generic
                               visualization
                               .plotData:          (struct) information
                                                   required to plot the data:
                                                   .plotDims: dims to plot
                                                   .projpt: projection points.
                                                            Can be vector or
                                                            cell. e.g.
                                                            {pi,'min'} means
                                                            project at pi for
                                                            first dimension,
                                                            take minimum
                                                            (union) for second
                                                            dimension
                               .sliceLevel:        (double) level set of
                                                   reachable set to visualize
                                                   (default is 0)
                               .holdOn:            (bool) leave whatever was
                                                    already on the figure?
                               .lineWidth:         (int) width of lines
                               .viewAngle:         (vector) [az,el] angles for
                                                   viewing plot
                               .camlightPosition:  (vector) location of light
                                                   source
                               .viewGrid:          (bool) view grid
                               .viewAxis:          (vector) size of axis
                               .xTitle:            (string) x axis title
                               .yTitle:            (string) y axis title
                               .zTitle:            (string) z axis title
                               .dtTime             How often you want to
                                                   update time stamp on title
                                                   of plot
                               .fontSize:          (int) font size of figure
                               .deleteLastPlot:    (bool) set to True to
                                                   delete previous plot
                                                   before displaying next one
                               .figNum:            (int) for plotting a
                                                   specific figure number
                               .figFilename:       (string) provide this to
                                                   save the figures (requires
                                                   export_fig package)
                               .initialValueSet:   (bool) view initial value
                                                   set
                               .initialValueFunction: (bool) view initial
                                                   value function
                               .valueSet:          (bool) view value set
                               .valueFunction:     (bool) view value function
                               .obstacleSet:       (bool) view obstacle set
                               .obstacleFunction:  (bool) view obstacle
                                                   function
                               .targetSet:         (bool) view target set
                               .targetFunction:    (bool) view target function
                               .plotColorVS0:      color of initial value set
                               .plotColorVF0:      color of initial value
                                                   function
                               .plotAlphaVF0:      transparency of initial
                                                   value function
                               .plotColorVS:       color of value set
                               .plotColorVF:       color of value function
                               .plotAlphaVF:       transparency of initial
                                                   value function
                               .plotColorOS:       color of obstacle set
                               .plotColorOF:       color of obstacle function
                               .plotAlphaOF:       transparency of obstacle
                                                   function
                               .plotColorTS:       color of target set
                               .plotColorTF:       color of target function
                               .plotAlphaTF:       transparency of target
                                                   function

     Outputs:
       data - solution corresponding to grid g and time vector tau
       tau  - list of computation times (redundant)
       extraOuts - This structure can be used to pass on extra outputs, for
                   example:
          .stoptau: time at which the reachable set contains the initial
                    state; tau and data vectors only contain the data till
                    stoptau time.

          .hVS0:   These are all figure handles for the appropriate
          .hVF0	set/function
          .hTS
          .hTF
          .hOS
          .hOF
          .hVS
          .hVF


     -------Updated 11/14/18 by Sylvia Herbert (sylvia.lee.herbert@gmail.com)

     Python July 29, 2021 (Lekan Molu)
    """

    ## Default parameters
    if numel(tau) < 2:
        error('Time vector must have at least two elements!')

    if not compMethod: compMethod = 'minVOverTime'

    if not extraArgs: extraArgs = Bundle(dict())

    extraOuts = [];
    quiet = False;
    lowMemory = False;
    keepLast = False;
    flipOutput = False;

    small = 1e-4;
    g = schemeData.grid;
    gDim = g.dim;
    # clns = [[':'] for i in range(gDim)]

    ## Backwards compatible

    if isfield(extraArgs, 'low_memory'):
        extraArgs.lowMemory = copy.deepcopy(extraArgs.low_memory)
        del extraArgs.low_memory;
        warn('we now use lowMemory instead of low_memory');


    if extraArgs.flip_output:
        extraArgs.flipOutput = copy.deepcopy(extraArgs.flip_output)
        del extraArgs.flip_output;
        logger.warning('we now use flipOutput instead of flip_output');


    if extraArgs.stopSet:
        extraArgs.stopSetInclude = copy.deepcopy(extraArgs.stopSet)
        del extraArgs.stopSet;
        logger.warning('we now use stopSetInclude instead of stopSet');


    if isfield(extraArgs, 'visualize'):
        if not isinstance (extraArgs.visualize, Bundle) and \
                    extraArgs.visualize:
            # remove visualize boolean
            del extraArgs.visualize;

            # reset defaults
            extraArgs.visualize = Bundle(dict(initialValueSet = 1,
                                                valueSet = 1))

        if isfield(extraArgs, 'RS_level'):
            extraArgs.visualize.sliceLevel = extraArgs.RS_level;
            del extraArgs.RS_level;
            logger.warning(f'we now use extraArgs.visualize.sliceLevel instead of {extraArgs.RS_level}')


        if isfield(extraArgs, 'plotData'):
            extraArgs.visualize.plotData = extraArgs.plotData;
            del extraArgs.plotData;
            logger.warning(f'we now use extraArgs.visualize.plotData instead of {extraArgs.plotData}');


        if isfield(extraArgs, 'deleteLastPlot'):
            extraArgs.visualize.deleteLastPlot = extraArgs.deleteLastPlot;
            del extraArgs.deleteLastPlot;
            logger.warning('we now use extraArgs.visualize.deleteLastPlot instead'
                            'of extraArgs.deleteLastPlot');


        if isfield(extraArgs, 'fig_num'):
            extraArgs.visualize.figNum = copy.deepcopy(extraArgs.fig_num)
            del extraArgs.fig_num;
            # logger.warning(['we now use extraArgs.visualize.figNum instead'\
            #     'of extraArgs.fig_num']);


        if isfield(extraArgs, 'fig_filename'):
            extraArgs.visualize.figFilename =copy.deepcopy( extraArgs.fig_filename)
            del extraArgs.fig_filename;
            # logger.warning(['we now use extraArgs.visualize.figFilename instead'\
            #     'of extraArgs.fig_filename']);


        if isfield(extraArgs, 'target'):
            # logger.warning(['you wrote extraArgs.target instead of' \
            #     'extraArgs.targetFunction'])
            extraArgs.targetFunction = copy.deepcopy(extraArgs.target)
            del extraArgs.target;


        if isfield(extraArgs, 'targets'):
            # logger.warning(['you wrote extraArgs.targets instead of' \
            #     'extraArgs.targetFunction'])
            extraArgs.targetFunction = copy.deepcopy(extraArgs.targets)
            del extraArgs.targets;


        if isfield(extraArgs, 'obstacle'):
            extraArgs.obstacleFunction = copy.deepcopy(extraArgs.obstacle)
            # logger.warning(['you wrote extraArgs.obstacle instead of' \
            #     'extraArgs.obstacleFunction'])
            del extraArgs.obstacle;


        if isfield(extraArgs, 'obstacles'):
            extraArgs.obstacleFunction = copy.deepcopy(extraArgs.obstacles)
            # logger.warning(['you wrote extraArgs.obstacles instead of' \
            #     'extraArgs.obstacleFunction'])
            del extraArgs.obstacles;




    ## Extract the information from extraargs
    # Quiet mode
    if isfield(extraArgs, 'quiet') and extraArgs.quiet:
        print('HJIPDE_solve running in quiet mode')
        quiet = True;


    # Low memory mode
    if isfield(extraArgs, 'lowMemory') and extraArgs.lowMemory:
        info('HJIPDE_solve running in low memory mode')
        lowMemory = True;

        # Save the output in reverse order
        if (extraArgs.flipOutput) and (extraArgs.flipOutput):
            flipOutput = True;




    # Only keep latest data (saves memory)
    if isfield(extraArgs, 'flipOutput') and extraArgs.keepLast:
        keepLast = True;


    #---Extract the information about obstacles--------------------------------
    obsMode = 'none';


    if isfield(extraArgs, 'obstacleFunction'):
        obstacles = extraArgs.obstacleFunction;

        # are obstacles moving or not?
        if numDims(obstacles) == gDim:
            obsMode = 'static';
            obstacle_i = obstacles;
        elif numDims(obstacles) == gDim + 1:
            obsMode = 'time-varying';
            obstacle_i = obstacles[0, ...];
        else:
            error('Inconsistent obstacle dimensions!')


        # We always take the max between the data and the obstacles
        # note that obstacles are negated.  That's because if you define the
        # obstacles using something like ShapeSphere, it sets it up as a
        # target. To make it an obstacle we just negate that.
        data0 = max(data0, -obstacle_i);


    #---Extract the information about targets----------------------------------
    targMode = 'none';

    if isfield(extraArgs, 'targetFunction'):
        targets = extraArgs.targetFunction;

        # is target function moving or not?
        if numDims(targets) == gDim:
            targMode = 'static';
            target_i = targets;
        elif numDims(targets) == gDim + 1:
            targMode = 'time-varying';
            target_i = targets[0, ...];
        else:
            error('Inconsistent target dimensions!')



    #---Stopping Conditions----------------------------------------------------

    # Check validity of stopInit if needed
    if isfield(extraArgs, 'stopInit'):
        if not isvector(extraArgs.stopInit) or gDim is not len(extraArgs.stopInit):
            error('stopInit must be a vector of length g.dim!')

    if isfield(extraArgs,'stopSetInclude') or isfield(extraArgs,'stopSetIntersect'):
        if isfield(extraArgs,'stopSetInclude'):
            stopSet = extraArgs.stopSetInclude
        else:
            stopSet = extraArgs.stopSetIntersect;

        if numDims(stopSet) != gDim or np.any(size(stopSet) != g.N.T):
            error('Inconsistent stopSet dimensions!')

        # Extract set of indices at which stopSet is negative
        setInds = stopSet[stopSet < 0];

        # Check validity of stopLevel if needed
        if isfield(extraArgs, 'stopLevel'):
            stopLevel = extraArgs.stopLevel;
        else:
            stopLevel = 0;



    ## Visualization
    if (isfield(extraArgs, 'visualize') and isinstance(extraArgs.visualize, Bundle))\
            or (isfield(extraArgs, 'makeVideo' and extraArgs.makeVideo):
        # Mark initial iteration, state that for the first plot we need
        # lighting
        timeCount = 0;
        needLight = True;


        #---Video Parameters---------------------------------------------------

        # # If we're making a video, set up the parameters
        # if extraArgs.makeVideo and extraArgs.makeVideo:
        #     if not extraArgs.videoFilename
        #         extraArgs.videoFilename = datetime.strftime(datetime.now, '%m-%d-%y_%H-%M') + '.mp4'];
        #
        #
        #     vout = VideoWriter(extraArgs.videoFilename,'MPEG-4');
        #     vout.Quality = 100;
        #     if extraArgs.frameRate
        #         vout.FrameRate = extraArgs.frameRate;
        #     else
        #         vout.FrameRate = 30;
        #
        #
        #     try
        #         vout.open;
        #     catch
        #         error('cannot open file for writing')




        #---Projection Parameters----------------------------------------------

        # Extract the information about plotData
        plotDims = ones(gDim, 1);
        projpt = [];
        if isfield(extraArgs.visualize, 'plotData'):
            # Dimensions to visualize
            # It will be an array of 1s and 0s with 1s means that dimension should
            # be plotted.
            plotDims = extraArgs.visualize.plotData.plotDims;

            # Points to project other dimensions at. There should be an entry point
            # corresponding to each 0 in plotDims.
            projpt = extraArgs.visualize.plotData.projpt;


        # Number of dimensions to be plotted and to be projected
        pDims = np.count_nonzero(plotDims);
        projDims = len(projpt);

        # Basic Checks
        if (pDims > 4):
            error('Currently plotting up to 3D is supported!');


        #---Defaults-----------------------------------------------------------

        if isfield(extraArgs, 'obstacleFunction') and isfield(extraArgs, 'visualize'):
            if not isfield(extraArgs.visualize, 'obstacleSet'):
                extraArgs.visualize.obstacleSet = 1

        if isfield(extraArgs, 'targetFunction') and isfield(extraArgs, 'visualize'):
            if not isfield(extraArgs.visualize, 'targetSet'):
                extraArgs.visualize.targetSet = 1;

        # grid on

        # Number of dimensions to be plotted and to be projected
        pDims = np.count_nonzero(plotDims);
        if np.char.isnumeric(projpt):
            projDims = len(projpt);
        else:
            projDims = gDim - pDims;

        # Set level set slice
        if isfield(extraArgs.visualize, 'sliceLevel'):
            sliceLevel = extraArgs.visualize.sliceLevel;
        else:
            sliceLevel = 0;


        # Do we want to see every single plot at every single time step, or
        # only the most recent one?
        if isfield(extraArgs.visualize, 'deleteLastPlot'):
            deleteLastPlot = extraArgs.visualize.deleteLastPlot
        else:
            deleteLastPlot = False;

        view3D = 0;

        # Project
        if (projDims == 0):
            gPlot = g;
            dataPlot = data0;
            if isfield(extraArgs, 'obstacleFunction'):
                obsPlot = obstacle_i;

            if isfield(extraArgs, 'targetFunction'):
                targPlot = target_i;
        else:
            # if projpt is a cell, project each dimensions separately. This
            # allows us to take the union/intersection through some dimensions
            # and to project at a particular slice through other dimensions.
            if isinstance(projpt, list):
                idx = np.where(plotDims==0)#[0]
                plotDimsTemp = ones(size(plotDims));
                gPlot = g;
                dataPlot = data0;
                if isfield(extraArgs, 'obstacleFunction'):
                    obsPlot = obstacle_i;

                if isfield(extraArgs, 'targetFunction'):
                    targPlot = target_i;

                for ii in range(len(idx[0])-1, 0, -1):
                    plotDimsTemp(idx[ii]) = 0;
                    if extraArgs.obstacleFunction:
                        _, obsPlot = proj(gPlot, obsPlot, np.logical_not(plotDimsTemp), projpt[ii]);

                    if extraArgs.targetFunction:
                        _, targPlot = proj(gPlot, targPlot, np.logical_not(plotDimsTemp), projpt[ii])

                    gPlot, dataPlot = proj(gPlot, dataPlot, np.logical_not(plotDimsTemp), projpt[ii])
                    plotDimsTemp = ones(1,gPlot.dim);
            else:
                gPlot, dataPlot = proj(g, data0, np.logical_not(plotDims), projpt);

                if extraArgs.obstacleFunction
                    _, obsPlot = proj(g, obstacle_i, np.logical_not(plotDims), projpt);

                if extraArgs.targetFunction
                    [_, targPlot] = proj(g, target_i, np.logical_not(plotDims), projpt);
        #---Initialize Figure--------------------------------------------------


        # Initialize the figure for visualization
        fig = plt.figure(figsize=winsize);
        fig.tight_layout()
        ax = fig.add_subplot(1, 1, 1)


        # Clear figure unless otherwise specified
        plt.clf()

        ax.grid('on')

        # Set defaults
        eAT_visSetIm = Bundle(dict(sliceDim = gPlot.dim,
                                    applyLight = False))
        if isfield(extraArgs.visualize, 'lineWidth'):
            eAT_visSetIm.LineWidth = extraArgs.visualize.lineWidth;
            eAO_visSetIm.LineWidth = extraArgs.visualize.lineWidth;
        else:
            eAO_visSetIm.LineWidth = 2;


        # If we're stopping once we hit an initial condition requirement, plot
        # said requirement
        if extraArgs.stopInit:
            projectedInit = extraArgs.stopInit(plotDims.astype(bool))
            if np.nonzero(plotDims) == 2:
                ax.plot(projectedInit[0], projectedInit[1], 'b*')
            elif np.nonzero(plotDims) == 3:
                ax.plot_wireframe(projectedInit[0], projectedInit[1], projectedInit[2], 'b*')



        ## Visualize Inital Value Function/Set

        #---Visualize Initial Value Set----------------------------------------
        if isfield(extraArgs.visualize, 'initialValueSet') and
                extraArgs.visualize.initialValueSet"

            if not isfield(extraArgs.visualize,'plotColorVS0'):
                extraArgs.visualize.plotColorVS0 = 'g';


            extraOuts.hVS0 = visSetIm(\
                gPlot, dataPlot, extraArgs.visualize.plotColorVS0,\
                sliceLevel, eAT_visSetIm);

            if isfield(extraArgs.visualize,'plotAlphaVS0'):
                extraOuts.hVS0.FaceAlpha = extraArgs.visualize.plotAlphaVS0;



        #---Visualize Initial Value Function-----------------------------------
        if isfield(extraArgs.visualize, 'initialValueFunction') and \
                extraArgs.visualize.initialValueFunction:

            # If we're making a 3D plot, mark so we know to view this at an
            # angle appropriate for 3D
            if gPlot.dim >= 2:
                view3D = 1;

            # Set up default parameters
            if isfield(extraArgs.visualize,'plotColorVF0'):
                extraArgs.visualize.plotColorVF0 = 'g';


            if not isfield(extraArgs.visualize,'plotAlphaVF0'):
                extraArgs.visualize.plotAlphaVF0 = .5;


            # Visualize Initial Value function (hVF0)
            extraOuts.hVF0= visFuncIm(gPlot,dataPlot,\
                extraArgs.visualize.plotColorVF0,\
                extraArgs.visualize.plotAlphaVF0);


        ## Visualize Target Function/Set

        #---Visualize Target Set-----------------------------------------------
        if isfield(extraArgs.visualize, 'targetSet')  \
                and extraArgs.visualize.targetSet:


            if not isfield(extraArgs.visualize,'plotColorTS'):
                extraArgs.visualize.plotColorTS = 'g';


            extraOuts.hTS = visSetIm(gPlot, targPlot, \
                extraArgs.visualize.plotColorTS, sliceLevel, eAT_visSetIm);

            if isfield(extraArgs.visualize,'plotAlphaTS')
                extraOuts.hTS.FaceAlpha = extraArgs.visualize.plotAlphaTS;



        #---Visualize Target Function------------------------------------------
        if isfield(extraArgs.visualize, 'targetFunction') and
                extraArgs.visualize.targetFunction:
            # If we're making a 3D plot, mark so we know to view this at an
            # angle appropriate for 3D
            if gPlot.dim >= 2
                view3D = 1

            # Set up default parameters
            if not isfield(extraArgs.visualize,'plotColorTF'):
                extraArgs.visualize.plotColorTF = 'g';


            if not isfield(extraArgs.visualize,'plotAlphaTF'):
                extraArgs.visualize.plotAlphaTF = .5;


            # Visualize Target function (hTF)
            extraOuts.hTF= visFuncIm(gPlot,targPlot,\
                extraArgs.visualize.plotColorTF,\
                extraArgs.visualize.plotAlphaTF);


        ## Visualize Obstacle Function/Set

        #---Visualize Obstacle Set---------------------------------------------
        if isfield(extraArgs.visualize, 'obstacleSet') \
                and extraArgs.visualize.obstacleSet:

            if not isfield(extraArgs.visualize,'plotColorOS'):
                extraArgs.visualize.plotColorOS = 'r';


            # Visualize obstacle set (hOS)
            extraOuts.hOS = visSetIm(gPlot, obsPlot, \
                extraArgs.visualize.plotColorOS, sliceLevel, eAO_visSetIm);


        #---Visualize Obstacle Function----------------------------------------
        if  isfield(extraArgs.visualize, 'obstacleFunction') \
                and extraArgs.visualize.obstacleFunction
            # If we're making a 3D plot, mark so we know to view this at an
            # angle appropriate for 3D
            if gPlot.dim >= 2
                view3D = 1;


            # Set up default parameters
            if not isfield(extraArgs.visualize,'plotColorOF'):
                extraArgs.visualize.plotColorOF = 'r';


            if not isfield(extraArgs.visualize,'plotAlphaOF'):
                extraArgs.visualize.plotAlphaOF = .5;


            # Visualize function
            [extraOuts.hOF]= visFuncIm(gPlot,-obsPlot,\
                extraArgs.visualize.plotColorOF,\
                extraArgs.visualize.plotAlphaOF);

        ## Visualize Value Function/Set
        #---Visualize Value Set Heat Map---------------------------------------
        if isfield(extraArgs.visualize, 'valueSetHeatMap') and
                extraArgs.visualize.valueSetHeatMap:
            maxVal = max(abs(data0(:)));
            clims = [-maxVal-1, maxVal+1];
            # fix this later
            extraOuts.hVSHeat = ax.imshow(np.hstack((gPlot.vs[0],gPlot.vs[1])),\
                                            cmap=dataPlot,vmin=clims[0], \
                                            vmax=clims[1], origin='lower');
            # if isfield(extraArgs.visualize,'colormap'):
            #     plt.colormap(extraArgs.visualize.colormap)
            # else:
            #     cmapAutumn = colormap('autumn');
            #     cmapCool = colormap('cool');
            #     cmap[:32,:] = cmapCool(64:-2:1,:);
            #     cmap[32:64,:] = cmapAutumn(64:-2:1,:);
            #     colormap(cmap);
            fig.colorbar()


        #---Visualize Value Set------------------------------------------------
        if isfield(extraArgs.visualize, 'valueSet') and extraArgs.visualize.valueSet:

            if not isfield(extraArgs.visualize,'plotColorVS'):
                extraArgs.visualize.plotColorVS = 'b'

            extraOuts.hVS = visSetIm(gPlot, dataPlot, extraArgs.visualize.plotColorVS, sliceLevel, eAT_visSetIm)


        #---Visualize Value Function-------------------------------------------
        if isfield(extraArgs.visualize, 'valueFunction') and  extraArgs.visualize.valueFunction:
            # If we're making a 3D plot, mark so we know to view this at an
            # angle appropriate for 3D
            if gPlot.dim >= 2:
                view3D = 1

            # Set up default parameters
            if not isfield(extraArgs.visualize,'plotColorVF'):
                extraArgs.visualize.plotColorVF = 'b'

            if not isfield(extraArgs.visualize,'plotAlphaVF'):
                extraArgs.visualize.plotAlphaVF = .5


            # Visualize Value function (hVF)
            extraOuts.hVF= visFuncIm(gPlot,dataPlot,\
                extraArgs.visualize.plotColorVF,\
                extraArgs.visualize.plotAlphaVF);



        ## General Visualization Stuff

        #---Set Angle, Lighting, axis, Labels, Title---------------------------

        # fix this later
        # # Set Angle
        # if pDims >2 or view3D or isfield(extraArgs.visualize, 'viewAngle'):
        #     if isfield(extraArgs.visualize, 'viewAngle'):
        #         view(extraArgs.visualize.viewAngle)
        #     else:
        #         view(30,10)
        #
        #
        #     # Set Lighting
        #     if needLight# and (gPlot.dim == 3)
        #         lighting phong
        #         c = camlight;
        #         #need_light = False;
        #
        #     if extraArgs.,camlightPosition')
        #         c.Position = extraArgs.visualize.camlightPosition;
        #     else
        #         c.Position = [-30 -30 -30];



        # # Grid and axis
        # if extraArgs.,viewGrid') and \
        #         ~extraArgs.visualize.viewGrid
        #     grid off
        #
        #
        # if extraArgs.,viewAxis')
        #     axis(extraArgs.visualize.viewAxis)
        #
        # axis square

        # Labels
        if isfield(extraArgs.visualize, 'xTitle'):
            ax.set_xlabel(extraArgs.visualize.xTitle)


        if isfield(extraArgs.visualize, 'yTitle'):
            ax.set_ylabel(extraArgs.visualize.yTitle)


        if extraArgs.,zTitle')
            ax.set_zlabel(extraArgs.visualize.zTitle)

        # ax.set_title(f't = {0} ')
        # gcf = plt.gcf()
        # set(gcf,'Color','white')
        #
        # if extraArgs.,fontSize')
        #     set(gca,'FontSize',extraArgs.visualize.fontSize)
        #
        #
        # if extraArgs.,lineWidth')
        #     set(gca,'LineWidth',extraArgs.visualize.lineWidth)


        # drawnow;

    ## Extract dynamical system if needed
    if isfield(schemeData, 'dynSys'):
        schemeData.hamFunc = genericHam;
        schemeData.partialFunc = genericPartial;


    stopConverge = False;
    if isfield(extraArgs, 'stopConverge'):
        stopConverge = extraArgs.stopConverge;
        if isfield(extraArgs, 'convergeThreshold'):
            convergeThreshold = extraArgs.convergeThreshold;
        else:
            convergeThreshold = 1e-5;


    ## SchemeFunc and SchemeData
    schemeFunc = termLaxFriedrichs;
    # Extract accuracy parameter o/w set default accuracy
    accuracy = 'medium' #'veryHigh'; # sylvia used veryHigh in her code
    if isfield(schemeData, 'accuracy'):
        accuracy = schemeData.accuracy;


    ## Numerical approximation functions
    dissType = 'global';
    schemeData.dissFunc, integratorFunc, schemeData.derivFunc = getNumericalFuncs(dissType, accuracy);

    # if we're doing minWithZero or zero as the comp method, actually implement
    # correctly using level set toolbox
    if compMethod=='minWithZero' or compMethod=='zero':
        schemeFunc = termRestrictUpdate;
        schemeData.innerFunc = termLaxFriedrichs;
        schemeData.innerData = schemeData;
        schemeData.positive = 0;


    ## Time integration
    integratorOptions = odeCFLset('factorCFL', 0.8, 'singleStep', 'on');

    startTime = cputime();

    ## Stochastic additive terms
    if extraArgs.addGaussianNoiseStandardDeviation:
        # We are taking all the previous scheme terms and adding noise to it
        # Save all the previous terms as the deterministic component in detFunc
        detFunc = schemeFunc;
        detData = schemeData;
        # The full computation scheme will include this added term so clear
        # out the schemeFunc so we can pack everything back in later with the
        # new stuff
        del schemeFunc, schemeData;

        # Create the Hessian term corresponding to white noise diffusion
        stochasticFunc = termTraceHessian;
        stochasticData.grid = g;
        stochasticData.L = extraArgs.addGaussianNoiseStandardDeviation.T;
        stochasticData.R = extraArgs.addGaussianNoiseStandardDeviation;
        stochasticData.hessianFunc = hessianSecond;

        # Add the (saved) deterministic terms and the (new) stochastic term
        # together into the complete scheme
        schemeFunc = termSum;
        schemeData.innerFunc = [ detFunc, stochasticFunc ]
        schemeData.innerData = [ detData, stochasticData ]


    ## Initialize PDE solution
    data0size = size(data0);

    if numDims(data0) == gDim:
        # New computation
        if keepLast:
            data = data0;
        elif lowMemory:
            data = data0.astype(np.float32)
        else:
            data = np.zeros((data0size[0:gDim] +(len(tau), )));
            data[clns[:], 0] = data0;


        istart = 2;
    elif numDims(data0) == gDim + 1:
        # Continue an old computation
        if keepLast:
            data = data0[data0size[-1], ...]; # check these indexing
        elif lowMemory:
            data = data0[data0size[-1], ...].astype(np.float32)
        else:
            data = np.zeros((data0size[:gDim]+(len(tau), )]))
            data[:data0size[-1], ...] = data0;


        # Start at custom starting index if specified
        if extraArgs.istart:
            istart = extraArgs.istart;
        else:
            istart = data0size[-1]+1;

    else:
        error('Inconsistent initial condition dimension!')



    if extraArgs.ignoreBoundary and extraArgs.ignoreBoundary:
        _, dataTrimmed = truncateGrid(g, data0, g.min+4*g.dx, g.max-4*g.dx);


    for i in range(istart, len(tau)):
        if not quiet:
            print('tau(i) = #f ', tau[i])

        ## Variable schemeData
        if extraArgs.SDModFunc:
            if extraArgs.SDModParams:
                paramsIn = extraArgs.SDModParams;
            else:
                paramsIn = [];


            schemeData = extraArgs.SDModFunc(schemeData, i, tau, data, obstacles, paramsIn);


        if keepLast:
            y0 = data;
        elif lowMemory:
            if flipOutput:
                y0 = data[ 0,...];
            else:
                y0 = data[size(data, g.dim), ...];


        else:
            y0 = data[i-1, ...]

        y = np.expand(y0.flatten(), 1);


        tNow = tau[i-1];

        ## Main integration loop to get to the next tau(i)
        while tNow < tau[i] - small:
            # Save previous data if needed
            if compMethod =='minVOverTime' or compMethod =='maxVOverTime':
                yLast = y;


            if not quiet:
                print('  Computing [{} {}] \n'.format(tNow, tau[i]))



            # Solve hamiltonian and apply to value function (y) to get updated
            # value function # integrator function is an odeCFL function
            tNow, y = integratorFunc(schemeFunc, [tNow, tau[i]], y, \
                                        integratorOptions, schemeData)

            if np.any(np.isnan(y)):
                logger.fatal(f'Nans encountered in the integrated result {y}')



            ## Here's where we do the min/max for BRTS or nothing for BRSs.  Note that
            #  if we're doing discounting there are two methods: Kene's and Jaime's.
            #  Kene requires that we do the compmethod inside of the discounting.
            #  Jaime's does not.  Thus why the if statements are a little funky.

            # 1. If not discounting at all OR not discounting using Kene's
            #    method, do normal compMethod first
            if ~extraArgs.discountMode || \
                    ~strcmp(extraArgs.discountMode, 'Kene')

                #   compMethod
                # - 'set' or 'none' to compute reachable set (not tube)
                # - 'zero' or 'minWithZero' to min Hamiltonian with zero
                # - 'minVOverTime' to do min with previous data
                # - 'maxVOverTime' to do max with previous data
                # - 'minVWithL' or 'minVWithTarget' to do min with targets
                # - 'maxVWithL' or 'maxVWithTarget' to do max with targets
                # - 'minVWithV0' to do min with original data (default)
                # - 'maxVWithV0' to do max with original data

                if strcmp(compMethod, 'zero') \
                        || strcmp(compMethod, 'set')\
                        || strcmp(compMethod, 'none')
                    # note: compMethod 'zero' is handled at the beginning of
                    # the code. compMethod 'set' and 'none' require no
                    # computation.
                elif strcmp(compMethod, 'minVOverTime') #Min over Time
                    y = min(y, yLast);
                elif strcmp(compMethod, 'maxVOverTime')
                    y = max(y, yLast);
                elif strcmp(compMethod, 'minVWithV0')#Min with data0
                    y = min(y,data0(:));
                elif strcmp(compMethod, 'maxVWithV0')
                    y = max(y,data0(:));
                elif strcmp(compMethod, 'maxVWithL')\
                        || strcmp(compMethod, 'maxVwithL') \
                        || strcmp(compMethod, 'maxVWithTarget')
                    if ~extraArgs.targetFunction
                        error('Need to define target function l(x)!')

                    if numDims(targets) == gDim
                        y = max(y, targets(:));
                    else
                        target_i = targets(clns[:], i);
                        y = max(y, target_i(:));

                elif strcmp(compMethod, 'minVWithL') \
                        || strcmp(compMethod, 'minVwithL') \
                        || strcmp(compMethod, 'minVWithTarget')
                    if ~extraArgs.targetFunction
                        error('Need to define target function l(x)!')

                    if numDims(targets) == gDim
                        y = min(y, targets(:));
                    else
                        target_i = targets(clns[:], i);
                        y = min(y, target_i(:));


                else
                    error('Check which compMethod you are using')



                # 2. If doing discounting but not using Kene's method, default
                #    to Jaime's method from ICRA 2019 paper
                if extraArgs.discountFactor and \
                        extraArgs.discountFactor and \
                        (~extraArgs.discountMode || \
                        strcmp(extraArgs.discountMode,'Kene'))
                    y = extraArgs.discountFactor*y;

                    if extraArgs.targetFunction
                        y = y + \
                            (1-extraArgs.discountFactor).*extraArgs.targets(:);
                    else
                        y = y + \
                            (1-extraArgs.discountFactor).*data0(:);





                # 3. If we are doing Kene's discounting from minimum discounted
                #    rewards paper, do that now and do compmethod with it
            elif extraArgs.discountFactor and \
                    extraArgs.discountFactor and \
                    extraArgs.discountMode and \
                    strcmp(extraArgs.discountMode,'Kene')

                if ~extraArgs.targetFunction
                    error('Need to define target function l(x)!')


                # move everything below 0
                maxVal = max(abs(extraArgs.targetFunction(:)));
                ytemp = y - maxVal;
                targettemp = extraArgs.targetFunction - maxVal;

                # Discount
                ytemp = extraArgs.discountFactor*ytemp;

                if strcmp(compMethod, 'minVWithL') \
                        || strcmp(compMethod, 'minVwithL') \
                        || strcmp(compMethod, 'minVWithTarget')
                    # Take min
                    ytemp = min(ytemp, targettemp(:));

                elif strcmp(compMethod, 'maxVWithL')\
                        || strcmp(compMethod, 'maxVwithL') \
                        || strcmp(compMethod, 'maxVWithTarget')
                    # Take max
                    ytemp = max(ytemp, targettemp(:));
                else
                    error('check your compMethod!')


                # restore height
                y = ytemp + maxVal;
            else
                # if this didn't work, check why
                error('check your discountFactor and discountMode')





            # "Mask" using obstacles
            if extraArgs.obstacleFunction
                if strcmp(obsMode, 'time-varying')
                    obstacle_i = obstacles(clns[:], i);

                y = max(y, -obstacle_i(:));



            # Update target function
            if extraArgs.targetFunction
                if strcmp(targMode, 'time-varying')
                    target_i = targets(clns[:], i);






        # Reshape value function
        data_i = reshape(y, g.shape);
        if keepLast
            data = data_i;
        elif lowMemory
            if flipOutput
                data = cat(g.dim+1, reshape(y, g.shape), data);
            else
                data = cat(g.dim+1, data, reshape(y, g.shape));


        else
            data(clns[:], i) = data_i;


        # If we're stopping once converged, print how much change there was in
        # the last iteration
        if stopConverge
            if extraArgs.')&\
                    extraArgs.ignoreBoundary
                [~, dataNew] = truncateGrid(\
                    g, data_i, g.min+4*g.dx, g.max-4*g.dx);
                change = max(abs(dataNew(:)-dataTrimmed(:)));
                dataTrimmed = dataNew;
                if ~quiet
                    print('Max change since last iteration: #f\n', change)

            else
                change = max(abs(y - y0(:)));
                if ~quiet
                    print('Max change since last iteration: #f\n', change)




        ## If commanded, stop the reachable set computation once it contains
        # the initial state.
        if extraArgs.stopInit
            initValue = eval_u(g, data_i, extraArgs.stopInit);
            if ~isnan(initValue) and initValue <= 0
                extraOuts.stoptau = tau(i);
                tau(i+1:) = [];

                if ~lowMemory and ~keepLast
                    data(clns[:], i+1:size(data, gDim+1)) = [];

                break



        ## Stop computation if reachable set contains a "stopSet"
        if exist('stopSet', 'var')
            dataInds = find(data_i(:) <= stopLevel);

            if extraArgs.stopSetInclude
                stopSetFun = @all;
            else
                stopSetFun = @any;


            if stopSetFun(ismember(setInds, dataInds))
                extraOuts.stoptau = tau(i);
                tau(i+1:) = [];

                if ~lowMemory and ~keepLast
                    data(clns[:], i+1:size(data, gDim+1)) = [];

                break



        ## Stop computation if we've converged
        if stopConverge and change < convergeThreshold

            if extraArgs.discountFactor and \
                    extraArgs.discountFactor and \
                    extraArgs.discountAnneal and \
                    extraArgs.discountFactor ~= 1

                if strcmp(extraArgs.discountAnneal, 'soft')
                    extraArgs.discountFactor = 1-((1-extraArgs.discountFactor)/2);

                    if abs(1-extraArgs.discountFactor) < .00005
                        extraArgs.discountFactor = 1;

                    print('\nDiscount factor: #f\n\n', \
                        extraArgs.discountFactor)
                elif strcmp(extraArgs.discountAnneal, 'hard') \
                        || extraArgs.discountAnneal==1
                    extraArgs.discountFactor = 1;
                    print('\nDiscount factor: #f\n\n', \
                        extraArgs.discountFactor)

            else
                extraOuts.stoptau = tau(i);
                tau(i+1:) = [];

                if ~lowMemory and ~keepLast
                    data(clns[:], i+1:size(data, gDim+1)) = [];

                break



        ## If commanded, visualize the level set.

        if (extraArgs.visualize and \
                (isstruct(extraArgs.visualize) || extraArgs.visualize == 1))\
                || (extraArgs.makeVideo and extraArgs.makeVideo)
            timeCount = timeCount + 1;

            # Number of dimensions to be plotted and to be projected
            pDims = nnz(plotDims);
            if isnumeric(projpt)
                projDims = length(projpt);
            else
                projDims = gDim - pDims;


            # Basic Checks
            if(length(plotDims) ~= gDim || projDims ~= (gDim - pDims))
                error('Mismatch between plot and grid dimensions!');



            #---Delete Previous Plot-------------------------------------------

            if deleteLastPlot
                if extraOuts.hOS and strcmp(obsMode, 'time-varying')
                    if iscell(extraOuts.hOS)
                        for hi = 1:length(extraOuts.hOS)
                            delete(extraOuts.hOS{hi})

                    else
                        delete(extraOuts.hOS);



                if extraOuts.hOF and strcmp(obsMode, 'time-varying')
                    if iscell(extraOuts.hOF)
                        for hi = 1:length(extraOuts.hOF)
                            delete(extraOuts.hOF{hi})

                    else
                        delete(extraOuts.hOF);


                if extraOuts.hTS and strcmp(targMode, 'time-varying')
                    if iscell(extraOuts.hTS)
                        for hi = 1:length(extraOuts.hTS)
                            delete(extraOuts.hTS{hi})

                    else
                        delete(extraOuts.hTS);



                if extraOuts.hTF and strcmp(targMode, 'time-varying')
                    if iscell(extraOuts.hTF)
                        for hi = 1:length(extraOuts.hTF)
                            delete(extraOuts.hTF{hi})

                    else
                        delete(extraOuts.hTF);


                if extraOuts.hVSHeat
                    if iscell(extraOuts.hVSHeat)
                        for hi = 1:length(extraOuts.hVSHeat)
                            delete(extraOuts.hVSHeat{hi})

                    else
                        delete(extraOuts.hVSHeat);


                if extraOuts.hVS
                    if iscell(extraOuts.hVS)
                        for hi = 1:length(extraOuts.hVS)
                            delete(extraOuts.hVS{hi})

                    else
                        delete(extraOuts.hVS);


                if extraOuts.hVF
                    if iscell(extraOuts.hVF)
                        for hi = 1:length(extraOuts.hVF)
                            delete(extraOuts.hVF{hi})

                    else
                        delete(extraOuts.hVF);





            #---Perform Projections--------------------------------------------

            # Project
            if projDims == 0
                gPlot = g;
                dataPlot = data_i;

                if strcmp(obsMode, 'time-varying')
                    obsPlot = obstacle_i;


                if strcmp(targMode, 'time-varying')
                    targPlot = target_i;

            else
                # if projpt is a cell, project each dimensions separately. This
                # allows us to take the union/intersection through some dimensions
                # and to project at a particular slice through other dimensions.
                if iscell(projpt)
                    idx = find(plotDims==0);
                    plotDimsTemp = ones(size(plotDims));
                    gPlot = g;
                    dataPlot = data_i;
                    if strcmp(obsMode, 'time-varying')
                        obsPlot = obstacle_i;


                    if strcmp(targMode, 'time-varying')
                        targPlot = target_i;


                    for ii = length(idx):-1:1
                        plotDimsTemp(idx(ii)) = 0;
                        if strcmp(obsMode, 'time-varying')
                            [~, obsPlot] = proj(gPlot, obsPlot, ~plotDimsTemp,\
                                projpt{ii});


                        if strcmp(targMode, 'time-varying')
                            [~, targPlot] = proj(gPlot, targPlot, ~plotDimsTemp,\
                                projpt{ii});


                        [gPlot, dataPlot] = proj(gPlot, dataPlot, ~plotDimsTemp,\
                            projpt{ii});
                        plotDimsTemp = ones(1,gPlot.dim);


                else
                    [gPlot, dataPlot] = proj(g, data_i, ~plotDims, projpt);

                    if strcmp(obsMode, 'time-varying')
                        [~, obsPlot] = proj(g, obstacle_i, ~plotDims, projpt);


                    if strcmp(targMode, 'time-varying')
                        [~, targPlot] = proj(g, obstacle_i, ~plotDims, projpt);








            ## Visualize Target Function/Set

            #---Visualize Target Set-----------------------------------------------
            if strcmp(targMode, 'time-varying') \
                    and extraArgs.,targetSet') \
                    and extraArgs.visualize.targetSet

                # Visualize obstacle set (hOS)
                extraOuts.hTS = visSetIm(gPlot, targPlot, \
                    extraArgs.visualize.plotColorTS, sliceLevel, eAT_visSetIm);

                if extraArgs.,'otAlphaTS')
                    extraOuts.hTS.FaceAlpha = extraArgs.visualize.plotAlphaTS;




            #---Visualize Target Function------------------------------------------
            if  strcmp(targMode, 'time-varying') \
                    and extraArgs.,targetFunction')\
                    and extraArgs.visualize.targetFunction

                # Visualize function
                [extraOuts.hTF]= visFuncIm(gPlot,targPlot,\
                    extraArgs.visualize.plotColorTF,\
                    extraArgs.visualize.plotAlphaTF);


            ## Visualize Obstacle Function/Set

            #---Visualize Obstacle Set-----------------------------------------
            if strcmp(obsMode, 'time-varying') \
                    and extraArgs.,obstacleSet') \
                    and extraArgs.visualize.obstacleSet

                # Visualize obstacle set (hOS)
                extraOuts.hOS = visSetIm(gPlot, obsPlot, \
                    extraArgs.visualize.plotColorOS, sliceLevel, eAO_visSetIm);

                if extraArgs.,'otAlphaOS')
                    extraOuts.hOS.FaceAlpha = extraArgs.visualize.plotAlphaOS;



            #---Visualize Obstacle Function------------------------------------
            if  strcmp(obsMode, 'time-varying') \
                    and extraArgs.visualize.obstacleFunction

                # Visualize function
                [extraOuts.hOF]= visFuncIm(gPlot,-obsPlot,\
                    extraArgs.visualize.plotColorOF,\
                    extraArgs.visualize.plotAlphaOF);

            ## Visualize Value Function/Set
            #---Visualize Value Set Heat Map-----------------------------------
            if extraArgs.,valueSetHeatMap') and\
                    extraArgs.visualize.valueSetHeatMap
                extraOuts.hVSHeat = imagesc(\
                    gPlot.vs{1},gPlot.vs{2},dataPlot,clims);
                #colorbar

            #---Visualize Value Set--------------------------------------------
            if extraArgs.,valueSet') and\
                    extraArgs.visualize.valueSet

                extraOuts.hVS = visSetIm(gPlot, dataPlot, \
                    extraArgs.visualize.plotColorVS, sliceLevel, eAT_visSetIm);


            if extraArgs.,'otAlphaVS')
                extraOuts.hVS.FaceAlpha = extraArgs.visualize.plotAlphaVS;

            #---Visualize Value Function---------------------------------------
            if extraArgs.,valueFunction') and \
                    extraArgs.visualize.valueFunction
                # Visualize Target function (hTF)
                [extraOuts.hVF]= visFuncIm(gPlot,dataPlot,\
                    extraArgs.visualize.plotColorVF,\
                    extraArgs.visualize.plotAlphaVF);



            #---Update Title---------------------------------------------------
            if ~extraArgs.,dtTime') and\
                    ~extraArgs.,convergeTitle')
                title(['t = ' num2str(tNow,'#4.2f') ' s'])
            elif extraArgs.,dtTime') and floor(\
                    extraArgs.visualize.dtTime/((tau()-tau(1))/length(tau))) \
                    == timeCount

                title(['t = ' num2str(tNow,'#4.2f') ' s'])
                timeCount = 0;
            elif extraArgs.')&\
                    extraArgs.stopConverge and\
                    extraArgs.,convergeTitle') and\
                    extraArgs.visualize.convergeTitle
                title(['t = ' num2str(tNow, '#4.2f') \
                    ' s, max change = ' num2str(change,'#4.4f')])
            else
                title(['t = ' num2str(tNow,'#4.2f') ' s'])

            drawnow;


            #---Save Video, Figure---------------------------------------------
            if extraArgs.makeVideo and extraArgs.makeVideo
                current_frame = getframe(gcf); #gca does just the plot
                writeVideo(vout,current_frame);


            if extraArgs.,figFilename')
                export_fig(sprintf('#s#d', \
                    extraArgs.visualize.figFilename, i), '-png')




        ## Save the results if needed
        if extraArgs.saveFilename
            if mod(i, extraArgs.saveFrequency) == 0
                ilast = i;
                save(extraArgs.saveFilename, 'data', 'tau', 'ilast', '-v7.3')




    ## Finish up
    if extraArgs.discountFactor and extraArgs.discountFactor
        extraOuts.discountFactor = extraArgs.discountFactor;


    Time = cputime;
    if ~quiet
        print('Total execution time #g seconds ', Time - startTime);


    if extraArgs.makeVideo and extraArgs.makeVideo
        vout.close

    return data, tau, extraOuts

    def getNumericalFuncs(dissType='global', accuracy='medium'):
        # Dissipation
        if (dissType == 'global'):
            dissFunc = artificialDissipationGLF
        # elif (dissType == 'local'):
        #     dissFunc = artificialDissipationLLF
        # elif (dissType == 'locallocal'):
        #     dissFunc = artificialDissipationLLLF
        # else:
            error('Unknown dissipation function #s'.format(dissType))


        # Accuracy
        # if accuracy is 'low':
        #     derivFunc = upwindFirstFirst
        #     integratorFunc = odeCFL1
        if accuracy is 'medium':
            derivFunc = upwindFirstENO2
            integratorFunc = odeCFL2
        # elif accuracy is 'high':
        #     derivFunc = upwindFirstENO3
        #     integratorFunc = odeCFL3
        # elif accuracy is 'veryHigh':
        #     derivFunc = upwindFirstWENO5
        #     integratorFunc = odeCFL3
        else:
            error('Unknown accuracy level #s'.format(accuracy))

        return dissFunc, integratorFunc, derivFunc
