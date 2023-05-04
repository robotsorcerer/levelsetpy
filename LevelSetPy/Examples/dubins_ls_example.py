__comment__     = "Solves the BRT of a P-E Dubins Vehicle in Relative Coordinates (Air3D Basic)."
__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__license__ 	= "Molux License"
__comment__ 	= "Evader at origin"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"
__date__ 		= "Nov. 2021"

import copy
import time
import logging
import argparse
import sys, os
import cupy as cp
import numpy  as np
from math import pi
import matplotlib.pyplot as plt
from cupyx.profiler import benchmark
import matplotlib.Gridspec as Gridspec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

from os.path import abspath, join, expanduser

from LevelSetPy.Utilities import *
from LevelSetPy.Visualization import *
from LevelSetPy.Grids import createGrid
from LevelSetPy.DynamicalSystems import DubinsVehicleRel
from LevelSetPy.InitialConditions import shapeCylinder
from LevelSetPy.SpatialDerivative import upwindFirstENO2
from LevelSetPy.ExplicitIntegration.Integration import odeCFL2, odeCFLset
from LevelSetPy.ExplicitIntegration.Dissipation import artificialDissipationGLF
from LevelSetPy.ExplicitIntegration.Term import termRestrictUpdate, termLaxFriedrichs
from LevelSetPy.Visualization import RCBRTVisualizer

parser = argparse.ArgumentParser(description='Hamilton-Jacobi Analysis')
parser.add_argument('--silent', '-si', action='store_false', help='silent debug print outs' )
parser.add_argument('--save', '-sv', action='store_false', help='save BRS/BRT at end of sim' )
parser.add_argument('--visualize', '-vz', action='store_false', help='visualize level sets?' )
parser.add_argument('--load_brt', '-lb', action='store_true', help='load saved brt?' )
parser.add_argument('--benchmark', '-bm', action='store_true', help='Benchmark this computation?' )
parser.add_argument('--stochastic', '-st', action='store_true', help='Run trajectories with stochastic dynamics?' )
parser.add_argument('--compute_traj', '-ct', action='store_false', help='Run trajectories with stochastic dynamics?' )
parser.add_argument('--verify', '-vf', action='store_true', default=True, help='visualize level sets?' )
parser.add_argument('--elevation', '-el', type=float, default=5., help='elevation angle for target set plot.' )
parser.add_argument('--direction', '-dr',  action='store_true',  help='direction to grow the level sets. Negative by default.' )
parser.add_argument('--azimuth', '-az', type=float, default=15., help='azimuth angle for target set plot.' )
parser.add_argument('--pause_time', '-pz', type=float, default=.3, help='pause time between successive updates of plots' )
args = parser.parse_args()
args.verbose = True if not args.silent else False

print(f'args:  {args}')

if not args.silent:
	logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
else:
	logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
# Turn off pyplot's spurious dumps on screen
logging.getLogger('matplotlib.font_manager').disabled = True
logger = logging.getLogger(__name__)

u_bound = 1
w_bound = 1
fontdict = {'fontsize':16, 'fontweight':'bold'}

def preprocessing():
	global u_bound, w_bound

	grid_min = expand(np.array((-.75, -1.25, -pi)), ax = 1)
	grid_max = expand(np.array((3.25, 1.25, pi)), ax = 1)
	pdDims = 2                      # 3rd dimension is periodic
	resolution = 100
	N = np.array(([[
					resolution,
					np.ceil(resolution*(grid_max[1, 0] - grid_min[1, 0])/ \
								(grid_max[0, 0] - grid_min[0, 0])),
					resolution-1
					]])).T.astype(int)
	grid_max[2, 0] *= (1-2/N[2,0])
	g = createGrid(grid_min, grid_max, N, pdDims)

	axis_align, center, radius = 2, np.zeros((3, 1)), 0.5
	value_init = shapeCylinder(g, axis_align, center, radius)

	return g, value_init

def main(args):
	# global params
	g, value_init = preprocessing()
	dubins_rel = DubinsVehicleRel(g, u_bound, w_bound)

	# after creating value function, make state space cupy objects
	g.xs = [cp.asarray(x) for x in g.xs]
	finite_diff_data = Bundle(dict(innerFunc = termLaxFriedrichs,
				innerData = Bundle({'grid':g, 'hamFunc': dubins_rel.hamiltonian,
				'partialFunc': dubins_rel.dissipation, 'dissFunc': artificialDissipationGLF,
				'CoStateCalc': upwindFirstENO2}), positive = args.direction, # direction to grow the updated level set
				))

	t_range = [0, 2.5]

	# Visualization paramters
	spacing = tuple(g.dx.flatten().tolist())
	init_mesh = implicit_mesh(value_init, level=0, spacing=spacing, edge_color='b', face_color='b')
	params = Bundle(
					{"grid": g,
					 'disp': True,
					 'labelsize': 16,
					 'labels': "Initial 0-LevelSet",
					 'linewidth': 2,
					 'data': value_init,
					 'elevation': args.elevation,
					 'azimuth': args.azimuth,
					 'mesh': init_mesh,
					 'init_conditions': False,
					 'pause_time': args.pause_time,
					 'level': 0, # which level set to visualize
					 'winsize': (16,9),
					 'fontdict': {'fontsize':18, 'fontweight':'bold'},
					 "savedict": Bundle({"save": False, "savename": "dint_basic.jpg",
										"savepath": "/opt/LevPy/Dubins"}),
					})
	args.spacing = spacing
	args.init_mesh = init_mesh; args.params = params
	os.makedirs(params.savedict.savepath) if not os.path.exists(params.savedict.savepath) else None

	if args.load_brt:
		args.save = False
		brt = np.load(join(params.savedict.savepath, "rcbrt.npz"))
	else:
		if args.visualize:
			viz = RCBRTVisualizer(params=args.params)
		t_steps = (t_range[1] - t_range[0]) / 10
		small = 100*eps
		options = Bundle(dict(factorCFL=0.95, stats='off', singleStep='off'))

		brt = [value_init]
		meshes, brt_time = [], []
		value_rolling = cp.asarray(copy.copy(value_init))

		t_now = t_range[0]
		gpu_time_buffer = []
		itr_start = cp.cuda.Event(); itr_end = cp.cuda.Event()
		while(t_range[1] - t_now > small * t_range[1]):
			itr_start.record()
			time_step = f"{t_now:.2f}/{t_range[-1]}"

			# Reshape data array into column vector for ode solver call.
			y0 = value_rolling.flatten()

			# How far to step?
			t_span = np.hstack([ t_now, min(t_range[1], t_now + t_steps) ])

			# Integrate a timestep.
			t, y, _ = odeCFL2(termRestrictUpdate, t_span, y0, odeCFLset(options), finite_diff_data)
			t_now = t

			# Get back the correctly shaped data array
			value_rolling = y.reshape(g.shape)

			if args.visualize:
				value_rolling_np = value_rolling.get()
				mesh=implicit_mesh(value_rolling_np, level=0, spacing=args.spacing,
									edge_color=None,  face_color='maroon')
				viz.update_tube(mesh, time_step, True)
				# store this brt
				brt.append(value_rolling_np); brt_time.append(t_now); meshes.append(mesh)

				if args.save:
					fig = plt.gcf()
					fig.savefig(join(params.savedict.savepath, rf"rcbrt_{t_now}.jpg"), bbox_inches='tight',facecolor='None')

			itr_end.record(); itr_end.synchronize()

			gpu_time_buffer.append(cp.cuda.get_elapsed_time(itr_start, itr_end)/1e3)
			info(f't: {time_step} | GPU time: {gpu_time_buffer[-1]:.4f} | Norm: {np.linalg.norm(y, 2):.2f}')

		if not args.load_brt:
			np.savez_compressed(join(params.savedict.savepath, "dubins.npz"), brt=np.asarray(brt), \
				meshes=np.asarray(meshes), brt_time=np.asarray(brt_time))

		info(f"Avg. local time: {sum(gpu_time_buffer)/len(gpu_time_buffer):.4f} secs")
		info(f"Total Time: {sum(gpu_time_buffer):.4f} secs")
		

if __name__ == '__main__':
	if args.benchmark:
		# Do not use python profiler: https://docs.cupy.dev/en/stable/user_guide/performance.html
		from cupyx.profiler import benchmark
		print(benchmark(main, (args,), n_repeat=20))
	else:
		main(args)