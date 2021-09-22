import time
from Utilities import *
from .data_proj import proj
from .compute_gradients import computeGradients
from matplotlib import pyplot as plt
from ExplicitIntegration import dynamics_RK4
from Visualization import vizLevelSet
from ValueFuncs import eval_u

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

		showDims = np.nonzero(extraArgs.projDim)[0]
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

	if visualize:
		f = plt.figure(figsize=(12, 7))
		f.tight_layout()
		fontdict = {'fontsize':12, 'fontweight':'bold'}
		plt.ion()
		# plt.rcParams['toolbar'] = 'None'
		# for key in plt.rcParams:
		# 	if key.startswith('keymap.'):
		# 		plt.rcParams[key] = ''

	while iter <= tauLength:
		# Determine the earliest time that the current state is in the reachable set
		# Binary search
		upper = tauLength
		lower = tEarliest

		tEarliest = lower #find_earliest_BRS_ind(g, data, dynSys.x, upper, lower)

		# BRS at current time
		BRS_at_t = data[:,:,:,tEarliest]

		# Visualize BRS corresponding to current trajectory point
		if visualize:
			ax = f.gca()
			# print(f'traj: {traj.shape} showDims {showDims}')
			ax.scatter(traj[showDims[0], iter], traj[showDims[1], iter], c='k')
			# plt.show()
			# print(f'g.vs in opt_traj: {[x.shape for x in g.vs]}')
			tStr = f't = {tau[iter]:.3f} tEarliest = {tau[tEarliest]:.3f}'
			ax.set_xlabel('X', fontdict=fontdict)
			ax.set_ylabel('Y', fontdict=fontdict)
			ax.grid('on')
			ax.set_title(tStr)
			ax.set_xlim(left=g.min[0], right=g.max[0])
			ax.set_ylim(g.min[1], g.max[1])
			# ax.set_color_cycle(['red', 'blue', 'black', 'green', 'magenta'])
			# print('BRS_at_t ', BRS_at_t.shape)
			geval = copy.deepcopy(g)
			g2D, data2D = proj(geval, BRS_at_t, hideDims, traj[hideDims,iter])
			# print(f'prj cot g.vs after eval: {[x.shape for x in g.vs]}')
			# print(f'prj cot geval.vs after eval: {[x.shape for x in geval.vs]}')
			try:
				ax.contour(g2D.xs[0], g2D.xs[1], data2D, levels=1, colors='g')
			except KeyboardInterrupt:
					plt.close('all')
			if isfield(extraArgs, 'fig_filename'):
				f.savefig(f'{extraArgs.fig_filename}_{iter}.png', bbox_inches='tight', dpi=79.0)
			# plt.show()
			# plt.cla()
			plt.pause(1)

		if tEarliest == tauLength:
			# Trajectory has entered the target
			break

		# Update trajectory
		Deriv, _, _ = computeGradients(geval, BRS_at_t, derivFunc=extraArgs.derivFunc)
		# print()
		# print(f'cot cg g_grad: {geval.shape},  geval: {[x.shape for x in geval.vs]}') #', datas: {[len(x) for x in Deriv]} {dynSys.x}')
		# print(f'cot cg g: {g.shape},  g: {[x.shape for x in g.vs]}') #', datas: {[len(x) for x in Deriv]} {dynSys.x}')


		for j in range(subSamples):
			# print(f'{j} b4: g.vs: {[x.shape for x in g.vs]}')
			# print(f'{j} dynSys.x : {dynSys.x.shape}')
			deriv = eval_u(copy.copy(g), Deriv, dynSys.x)
			# print(f'{j} after: g: {[x.shape for x in g.vs]}')
			# print(f'ss fl  g_eval: {[x.shape for x in g_eval.vs]}') #', datas: {[len(x) for x in Deriv]} {dynSys.x}')
			# print(f'{j} ss fl  g: {[x.shape for x in g.vs]}') #', datas: {[len(x) for x in Deriv]} {dynSys.x}')

			u = dynSys.get_opt_u(tau[tEarliest], deriv, uMode, dynSys.x)
			if dMode or var:
				d = dynSys.get_opt_v(tau[tEarliest], deriv, dMode, dynSys.x)
			else:
				d = dynSys.get_opt_v(tau[tEarliest], deriv, None, dynSys.x)
			# integrate the dynamics
			print(f'{j}: dynSys.x.T first: {dynSys.x.T}')
			xt = dynSys.update_state(u, dtSmall, dynSys.x, d)

			# print(f'{j}: dynSys.x.T: {dynSys.x.T}')
		# Record new point on nominal trajectory
		iter += 1
		traj[:,iter] = dynSys.x.squeeze()

	# Delete unused indices
	traj = traj[:,:iter]
	traj_tau = tau[:iter-1]

	return  traj, traj_tau
