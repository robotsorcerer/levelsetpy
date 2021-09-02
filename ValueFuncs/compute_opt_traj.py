from Utilities import *
from .data_proj import proj
from .compute_gradients import computeGradients
from matplotlib import pyplot as plt
from ExplicitIntegration import dynamics_RK4
from Visualization import vizLevelSet

def computeOptTraj(g, data, tau, dynSys, extraArgs=Bundle({})):
    """
     [traj, traj_tau] = computeOptTraj(g, data, tau, dynSys, extraArgs)
       Computes the optimal trajectories given the optimal value function
       represented by (g, data), associated time stamps tau, dynamics given in
       dynSys.

     Inputs:
       g, data - grid and value function
       tau     - time stamp (must be the same length as size of last dimension of
                             data)
       dynSys  - dynamical system object for which the optimal path is to be
                 computed
       extraArgs
         .uMode        - specifies whether the control u aims to minimize or
                         maximize the value function
         .dMode        - same for disurbance
         .visualize    - set to true to visualize results
         .fig_num:   List if you want to plot on a specific figure number
         .projDim      - set the dimensions that should be projected away when
                         visualizing
         .fig_filename - specifies the file name for saving the visualizations
    """

    # Default parameters
    uMode = 'min';
    visualize = False;
    subSamples = 4;

    if isfield(extraArgs, 'uMode'):
        uMode = extraArgs.uMode;

    if isfield(extraArgs, 'dMode'):
        dMode = extraArgs.dMode;

    # Visualization
    if isfield(extraArgs, 'visualize') and extraArgs.visualize:
        visualize = extraArgs.visualize;

        showDims = np.nonzero(extraArgs.projDim);
        hideDims = np.logical_not(extraArgs.projDim)

        # if isfield(extraArgs,'fig_num'):
        #     f = figure(extraArgs.fig_num);
        # else:
        f = plt.figure()

    if isfield(extraArgs, 'subSamples'):
        subSamples = extraArgs.subSamples;

    clns = [':' for i in range(g.dim)]

    if np.any(np.diff(tau, n=1, axis=0)) < 0:
        error('Time stamps must be in ascending order!')

    # Time parameters
    iter = 0;
    tauLength = len(tau);
    dtSmall = (tau[1]- tau[0])/subSamples;
    # maxIter = 1.25*tauLength;

    # Initialize trajectory
    traj = np.empty((g.dim, tauLength))
    traj.fill(np.nan)
    traj[:,0] = dynSys.x;
    tEarliest = 0;

    while iter <= tauLength:
        # Determine the earliest time that the current state is in the reachable set
        # Binary search
        upper = tauLength;
        lower = tEarliest;

        tEarliest = lower; #find_earliest_BRS_ind(g, data, dynSys.x, upper, lower);

        # BRS at current time
        BRS_at_t = data[tEarliest, ...];

        # Visualize BRS corresponding to current trajectory point
        if visualize:
            plt.plot(traj(showDims(1), iter), traj(showDims(2), iter), 'k.')
            g2D, data2D = proj(g, BRS_at_t, hideDims, traj[hideDims,iter]);
            vizLevelSet(g2D, data2D); # write this
            tStr = logger.info('t = {.3f}; tEarliest = {.3f}'.format(tau[iter], tau[tEarliest]));
            plt.title(tStr)

            if isfield(extraArgs, 'fig_filename'):
                plt.savefig(f'{extraArgs.fig_filename}_{iter}.png', dpi=79.0)

        if tEarliest == tauLength:
            # Trajectory has entered the target
            break

        # Update trajectory
        Deriv = computeGradients(g, BRS_at_t);
        N_Sim = 10  # lenth of integration steps
        X_OL = list() # to store open loop control law on evolved states

        for j in range(subSamples):
            deriv = eval_u(g, Deriv, dynSys.x);
            u = dynSys.get_opt_u(deriv, uMode, tau[tEarliest], dynSys.x);
            if dMode or var:
                d = dynSys.get_opt_v(deriv, dMode, tau[tEarliest], dynSys.x);
            else:
                d = dynSys.get_opt_v(deriv, tau[tEarliest], dynSys.x)
            xt = dynSys.update_dynamics(u, dtSmall, dynSys.x, d);

            # integrate the dynamics
            for t in range(N_sim):
                xt = dynamics_RK4(dynSys.update_dynamics, xt, u, d)
                X_OL.append(np.array(xt))
        # update dynamical systems state
        dynSys.x = X_OL[-1]
        dynSys.xhist = np.concatenate((dynsSus.xhist, dynSys.x), 1)
        dynSys.uhist = np.concatenate((dynsSus.uhist, u), 1)

        # Record new point on nominal trajectory
        iter += 1
        traj[:,iter] = dynSys.x

    # Delete unused indices
    traj = traj[:,:iter+1]
    traj_tau = tau[:iter]

    return  traj, traj_tau
