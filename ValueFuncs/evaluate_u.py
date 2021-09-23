from Utilities import *
from BoundaryCondition import addGhostPeriodic
from .augment_periodic import augmentPeriodicData
from ValueFuncs import *
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator

def eval_u(gs, datas, xs, interp_method='linear'):
	"""
	 v = eval_u(g, datas, x)
	   Computes the interpolated value of the value function data at the
	   states xs

	 Inputs:
	   Option 1: Single grid, single value function, multiple states
		 gs    - a single grid structure
		 datas - a single matrix (look-up table) representing the value
				 function
		 xs    - set of states each row is a state

	   Option 2: Single grid, multiple value functions, single state
		 gs    - a single grid structure
		 datas - a list of matrices representing the value function
		 xs    - a single state

	   Option 3: Multiple grids, value functions, and states. The number of
				 grids, value functions, and states must be equal under this
				 option
		 gs    - list containing multiple grid structures
		 datas - list containing multiple matrices of value functions
		 xs    - list containing multiple states
	Output:
		Returns the value function, v

	 Lekan Molux, 2021-08-09
	 """
	# for parti-games, the center of a cell may return a 1-D array
	if xs.ndim < 2: xs = np.expand_dims(xs, 0)
	if isinstance(gs, Bundle) and isinstance(datas, np.ndarray) and len(xs.shape)>=2:
		# Option 1
		v = eval_u_single(gs, datas, xs, interp_method)
	elif isinstance(gs, Bundle) and iscell(datas) and isvector(xs):
		# Option 2
		v = [np.nan for i in range(len(datas))]
		for i in range(len(datas)):
			# print(f'gs[{i}]: {gs.shape}')
			v[i] = eval_u_single(gs, datas[i], xs, interp_method)
			# print(f'gs[{i}] aft: {gs.shape}')
		v = np.asarray(v, order=ORDER_TYPE)
	elif iscell(gs) and iscell(datas) and iscell(xs):
		# Option 3
		v = cell(len(gs), 1)
		for i in range(len(gs)):
			v[i] = eval_u_single(gs[i], datas[i], xs[i], interp_method)
		v = np.asarray(v, order=ORDER_TYPE)
	else:
		error('Unrecognized combination of input data types!')

	return v

def  eval_u_single(g, data, x, interp_method):
	"""
	 v = eval_u_single(g, data, x)
	   Computes the interpolated value of a value function data at state x

	 Inputs:
	   g       - grid
	   data    - implicit function describing the set
	   x       - points to check each row is a point

	 OUTPUT
	   v:  value at points x

	 Python Ver: August 9, 2021

	"""
	# If the number of columns does not match the grid dimensions, try taking
	# transpose
	if size(x, 1) != g.dim:
	  x = x.T

	geval, dataOld = copy.deepcopy(g), copy.copy(data)
	g, data = augmentPeriodicData(geval, dataOld)

	# Dealing with periodicity
	for i in range(g.dim):
		if (isfield(g, 'bdry') and id(g.bdry[i])==id(addGhostPeriodic)):
			# Map input points within grid bounds
			period = max(g.vs[i]) - min(g.vs[i])

			i_above_bounds = x[:,i] > max(g.vs[i])
			while np.any(i_above_bounds):
				x[i_above_bounds, i] -= period
				i_above_bounds = x[:,i] > max(g.vs[i])

			i_below_bounds = x[:,i] < min(g.vs[i])
			while np.any(i_below_bounds):
				x[i_below_bounds, i] += period
				i_below_bounds = x[:,i] < min(g.vs[i])

	# Interpolate
	data_tup = [x.squeeze() for x in g.vs]
	# print(f'in eval data_tup: {[x.shape for x in data_tup]} data: {data.shape} dataOld: {dataOld.shape}')
	# print(f'in eval geval: {[x.shape for x in geval.vs]}')
	interp_func = RegularGridInterpolator(data_tup, data)
	if len(x)==len(g.vs):
		eval_pts = x.squeeze()
	else:
		eval_pts = [xx.squeeze() for xx in [x]*len(g.vs)]

	# print(f'eval_pts: {[x.shape for x in eval_pts]}')
	v = interp_func(eval_pts)

	# print(f'v: {v}')
	return v.take(0)
