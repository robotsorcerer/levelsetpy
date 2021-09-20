import time
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
    uMode = 'min'
    visualize = False
    subSamples = 4

    if isfield(extraArgs, 'uMode'):
        uMode = extraArgs.uMode

    if isfield(extraArgs, 'dMode'):
        dMode = extraArgs.dMode

    # Visualization
    if isfield(extraArgs, 'visualize') and extraArgs.visualize:
        visualize = extraArgs.visualize

        showDims = np.nonzero(extraArgs.projDim)
        hideDims = np.logical_not(extraArgs.projDim).squeeze()
        # print(f'showDims: {showDims}, hideDims: {hideDims}')
        f = plt.figure(figsize=(12, 7))
        ax = f.add_subplot(111)
        fontdict = {'fontsize':12, 'fontweight':'bold'}

    if isfield(extraArgs, 'subSamples'):
        subSamples = extraArgs.subSamples

    if np.any(np.diff(tau, n=1, axis=0)) < 0:
        error('Time stamps must be in ascending order!')

    # Time parameters
    iter = 0
    tauLength = len(tau)
    dtSmall = (tau[1]- tau[0])/subSamples
    # maxIter = 1.25*tauLength

    # Initialize trajectory
    traj = np.empty((g.dim, tauLength))
    traj.fill(np.nan)
    traj[:,0] = dynSys.x
    tEarliest = 0

    while iter <= tauLength:
        # Determine the earliest time that the current state is in the reachable set
        # Binary search
        upper = tauLength
        lower = tEarliest

        tEarliest = lower #find_earliest_BRS_ind(g, data, dynSys.x, upper, lower)

        # BRS at current time
        BRS_at_t = data[tEarliest, ...]

        # Visualize BRS corresponding to current trajectory point
        if visualize:
            ax.plot(traj[showDims[0], iter], traj[showDims[1], iter], color='k', linestyle='-.')
            # plt.show()
            # print(f'g.vs in opt_traj: {[x.shape for x in g.vs]}')
            g2D, data2D = proj(g, BRS_at_t, hideDims, traj[hideDims,iter])
            tStr = f't = {tau[iter]:.3f} tEarliest = {tau[tEarliest]:.3f}'
            ax.contour(g2D.xs[0], g2D.xs[1], data2D, levels=1, colors='g')
            ax.set_xlabel('X', fontdict=fontdict)
            ax.set_ylabel('Y', fontdict=fontdict)
            ax.grid('on')
            ax.set_title(tStr)

            if isfield(extraArgs, 'fig_filename'):
                f.savefig(f'{extraArgs.fig_filename}_{iter}.png', dpi=79.0)
            plt.show()
            time.sleep(.3)

        if tEarliest == tauLength:
            # Trajectory has entered the target
            break

        # Update trajectory
        Deriv = computeGradients(g, BRS_at_t)

        for j in range(subSamples):
            deriv = eval_u(g, Deriv, dynSys.x)
            u = dynSys.get_opt_u(tau[tEarliest], deriv, uMode, dynSys.x)
            if dMode or var:
                d = dynSys.get_opt_v(tau[tEarliest], deriv, dMode, dynSys.x)
            else:
                d = dynSys.get_opt_v(tau[tEarliest], deriv, None, dynSys.x)
            # integrate the dynamics
            xt = dynSys.update_state(u, dtSmall, dynSys.x, d)

        # Record new point on nominal trajectory
        iter += 1
        traj[:,iter] = dynSys.x

    # Delete unused indices
    traj = traj[:,:iter]
    traj_tau = tau[:iter-1]

    return  traj, traj_tau
