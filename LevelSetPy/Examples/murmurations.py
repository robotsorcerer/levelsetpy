__all_ = ["visualize_init_avoid_tube", "get_avoid_brt", "get_flock"]

__comment__     = "Solves the Complex Backward Reach Avoid Tubes for a Murmuration of Starlings."
__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__license__ 	= "Molux License"
__comment__ 	= "Murmutations of Starlings"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Ongoing"
__date__ 		= "Nov. 2021"




import copy
import time
import h5py
import logging
import argparse
import sys, os
import random
import cupy as cp
import numpy  as np
from math import pi
import numpy.linalg as LA
import scipy.linalg as spla
import matplotlib as mpl
# mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
from datetime import datetime

from os.path import abspath, join, dirname, expanduser
from LevelSetPy.Grids import *
from LevelSetPy.Utilities import *
from LevelSetPy.Visualization import *
from LevelSetPy.DynamicalSystems import *
from LevelSetPy.BoundaryCondition import *
from LevelSetPy.InitialConditions import *
from LevelSetPy.SpatialDerivative import *
from LevelSetPy.ExplicitIntegration import *
from LevelSetPy.Visualization import RCBRTVisualizer

parser = argparse.ArgumentParser(description='Murmurations\' Hamilton-Jacobi Analysis')
parser.add_argument('--flock_num', '-fn', type=int, default=0, help='Which flock\'s brat to optimize?' )
parser.add_argument('--silent', '-si', action='store_false', help='Silent debug print outs' )
parser.add_argument('--save', '-sv', action='store_false', default=True, help='Save BRS/BRT at end of sim' )
parser.add_argument('--out_dir', '-od', type=str, default="./data", help='Save value function to such and such folder' )
parser.add_argument('--visualize', '-vz', action='store_true', help='Visualize level sets?' )
parser.add_argument('--flock_payoff', '-sp', action='store_false', default=False, help='Visualize individual payoffs within a flock?' )
parser.add_argument('--resume', '-rz', type=str, help='Resume BRAT optimization from a previous iteration?' )
parser.add_argument('--mode', '-md', default='single', type=str, help='Stitch BRATs together? <stitch|single>' )
parser.add_argument('--load_brt', '-lb', action='store_true', help='Load saved brt?' )
parser.add_argument('--verify', '-vf', action='store_true', default=False, help='Verify a trajectory?' )
parser.add_argument('--elevation', '-el', type=float, default=25., help='Elevation angle for target set plot.' )
parser.add_argument('--azimuth', '-az', type=float, default=45., help='Azimuth angle for target set plot.' )
parser.add_argument('--pause_time', '-pz', type=float, default=.3, help='Pause time between successive updates of plots' )
args = parser.parse_args()
args.verbose = True if not args.silent else False

print(f'args:  {args}')

if not args.silent:
	logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
else:
	logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

# Turn off pyplot's spurious dumps on screen
logging.getLogger('matplotlib.font_manager').disabled = True
logger = logging.getLogger(__name__)

u_bound = 1
w_bound = 1 
fontdict = {'fontsize':16, 'fontweight':'bold'}

def visualize_init_avoid_tube(flock, save=True, fname=None, title=''):
	"""
		For a flock, whose mesh has been precomputed,
		visualize the initial backward avoid tube.
	"""

	fontdict = {'fontsize':16, 'fontweight':'bold'}
	mesh_bundle = flock.mesh_bundle

	fig = plt.figure(1, figsize=(16,9), dpi=100)
	ax = plt.subplot(111, projection='3d')
	ax.add_collection3d(mesh_bundle.mesh)


	xlim = (mesh_bundle.verts[:, 0].min(), mesh_bundle.verts[:,0].max())
	ylim = (mesh_bundle.verts[:, 1].min(), mesh_bundle.verts[:,1].max())
	zlim = (mesh_bundle.verts[:, 2].min(), mesh_bundle.verts[:,2].max())

	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
	ax.set_zlim(zlim)

	ax.grid('on')
	ax.tick_params(axis='both', which='major', labelsize=10)

	ax.set_xlabel(rf'x$_1^{flock.label}$ (m)', fontdict=fontdict)
	ax.set_ylabel(rf'x$_2^{flock.label}$ (m)', fontdict=fontdict)
	ax.set_zlabel(rf'$\omega^{flock.label} (deg)$',fontdict=fontdict)

	if title:
		ax.set_title(title, fontdict=fontdict)
	else:
		ax.set_title(f'Flock {flock.label}\'s ({flock.N} Agents) Payoff.', fontdict=fontdict)
	ax.view_init(azim=-45, elev=30)

	if save:
		plt.savefig(fname, bbox_inches='tight',facecolor='None')

def get_avoid_brt(flock, compute_mesh=True, color='crimson'):
	"""
		Get the avoid BRT for this flock. That is, every bird
		within a flock must avoid one another.

		Parameters:
		==========
		.flock: This flock of vehicles.
		.compute_mesh: compute mesh of local payoffs for each bird in this flock?
	"""
	for vehicle in flock.vehicles:
		# make the radius of the target setthe turn radius of this vehicle
		vehicle.payoff = shapeCylinder(flock.grid, 2, center=flock.update_state(vehicle.cur_state), \
										radius=vehicle.payoff_width)
	"""
		Now compute the overall payoff for the flock
	   	by taking a union of all the avoid sets.
	"""
	flock.payoff = shapeUnion([veh.payoff for veh in flock.vehicles])
	if compute_mesh:
		spacing=tuple(flock.grid.dx.flatten().tolist())
		flock.mesh_bundle = implicit_mesh(flock.payoff, 0, spacing,edge_color=None, face_color=color)

	return flock

def get_flock(gmin, gmax, num_points, num_agents, init_xyzs, label,\
				periodic_dims=2, reach_rad=.2, avoid_rad=.3,
				base_path='', save=True, color='blue'):
	"""
		Params
		======
		gmin: minimum bounds of the grid
		gmax: maximum bounds of the grid
		num_points: number of points on the grid
		num_agents: number of agents in this flock
		init_xyzs: initial positions of the individuals in this flock
		label: topological label of this flock among all flocks
		periodic_dims: periodic dimensions (usually theta: see)
		reach_rad: reach for external disturbance
		avoid_rad: avoid radius for topological interactions
	"""
	global u_bound, w_bound

	assert gmin.ndim==2, 'gmin must be of at least 2 dims'
	assert gmax.ndim==2, 'gmax must be of at least 2 dims'

	gmin = to_column_mat(gmin)
	gmax = to_column_mat(gmax)

	grid = createGrid(gmin, gmax, num_points, periodic_dims)

	vehicles = [Bird(grid, u_bound, w_bound, np.expand_dims(init_xyzs[i], 1), random.random(), \
					center=np.zeros((3,1)), neigh_rad=3, label=i+1, init_random=False) \
					for i in range(num_agents)]
	flock = Flock(grid, vehicles, label=label, reach_rad=.2, avoid_rad=.3)
	get_avoid_brt(flock, compute_mesh=True, color=color)

	if args.visualize and args.flock_payoff:
		visualize_init_avoid_tube(flock, save, fname=join(base_path, f"flock_{flock.label}.jpg"))
		plt.show()

	return flock


def main(args):
	# global params
	gmin = np.asarray([[-1.5, -1.5, -np.pi]]).T
	gmax = np.asarray([[1.5, 1.5, np.pi] ]).T

	H         = 0.4
	H_STEP    = 0.05
	neigh_rad = 0.4
	reach_rad = .2
	mul_factors = [1, 1.1, -1.1, 1.5, -1.5, 2.0, -1.8]
	num_agents = len(mul_factors)

	INIT_XYZS = np.array([[neigh_rad*np.cos((i/6)*np.pi/4), neigh_rad*np.sin((i/6)*np.pi/4), H+i*H_STEP] for i in range(num_agents)])

	# color thingy
	color = iter(plt.cm.inferno_r(np.linspace(.25, 1, num_agents)))

	# save shenanigans
	base_path = join("/opt/LevPy/Murmurations", 
						f"flock_{args.flock_num}",
						datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'))
	

	if strcmp(args.mode, 'single'):
		assert args.flock_num<num_agents and args.flock_num>=0, "Unknown flock number entered."
		scaling_factor = mul_factors[args.flock_num]	
		num_agents -= (args.flock_num%2)
		flock = get_flock(gmin, gmax, 101, num_agents, scaling_factor*INIT_XYZS, label=num_agents,\
						periodic_dims=2, reach_rad=.2, avoid_rad=.3, base_path=base_path, \
						color=next(color))

		finite_diff_data = Bundle(dict(innerFunc = termLaxFriedrichs,
					innerData = Bundle({'grid':flock.grid,
						'hamFunc': flock.hamiltonian,
						'partialFunc': flock.dissipation,
						'dissFunc': artificialDissipationGLF,
						'CoStateCalc': upwindFirstWENO5a, 
						}),
						positive = True,  # direction to grow the updated level set
					))

		# Visualization paramters
		params = Bundle(
						{"grid": flock.grid,
						'disp': True,
						'labelsize': 16,
						'labels': "Initial 0-LevelSet",
						'linewidth': 2,
						'elevation': 10,
						'azimuth': 5,
						'mesh': flock.mesh_bundle,
						'pause_time': args.pause_time,
						'title': f'Initial BRT. Flock with {flock.N} agents.',
						'level': 0, # which level set to visualize
						'winsize': (16,9),
						'fontdict': {'fontsize':18, 'fontweight':'bold'},
						"savedict": Bundle({"save": True,
										"savename": "murmur",
										"savepath": join("/opt/LevPy/Murmurations")
						})
				})

		os.makedirs(params.savedict.savepath) if not os.path.exists(params.savedict.savepath) else None
		if args.load_brt:
			args.save = False
			brt = np.load(join(params.savedict.savepath, "murmurations.npz"))
		else:
			if args.visualize:
				viz = RCBRTVisualizer(params=params)
			t_range = [0, 100]
			t_steps = (t_range[1] - t_range[0]) / 10000
			small = 100*eps

			# Loop through t_range (subject to a little roundoff).
			t_now = t_range[0]
			start_time = time.perf_counter()
			itr_start = cp.cuda.Event()
			itr_end = cp.cuda.Event()

			value_rolling = cp.asarray(copy.copy(flock.payoff))

			colors = iter(plt.cm.ocean(np.linspace(.25, 2, 100)))
			color = next(colors)
			options = Bundle(dict(factorCFL=0.7, stats='on', singleStep='on'))

			# murmur flock savename
			if args.resume:
				savename = args.resume

				# look up the last time index, load the brt, and advance the integration
				with h5py.File(savename, 'r+') as df:
					last_key = [key for key in df['value']][-1]
					value_rolling = np.asarray(df[f"value/{last_key}"])
					value_rolling = cp.asarray(value_rolling)
					t_now = float(last_key.split(sep='_')[-1])
			else:
				savename = join(params.savedict.savepath, rf"murmurations_flock_{args.flock_num:0>2}_{datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M')}.hdf5")
				if os.path.exists(savename):
					os.remove(savename)

			spacing = flock.grid.dx.flatten()

			with h5py.File(savename, 'a') as h5file:
				if not args.resume:
					# save spacing now
					h5file.create_dataset(f'value/spacing', data=spacing, compression="gzip")
			times, norm_tracker = [], []
			start_time = cputime()
			
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
				t, y, _ = odeCFL3(termRestrictUpdate, t_span, y0, odeCFLset(options), finite_diff_data)
				t_now = t

				# Get back the correctly shaped data array
				value_rolling = y.reshape(flock.grid.shape)

				# track the evolution of the growth of the level set so that when the norm stops decreasing, 
				# we terminate the game
				norm_tracker.append(LA.norm(y, 2))

				if args.visualize:
					value_rolling_np = value_rolling.get()
					mesh_bundle=implicit_mesh(value_rolling_np, level=0, \
											  spacing=tuple(spacing.tolist()), \
											  edge_color=None,  face_color=next(colors))
					viz.update_tube(mesh_bundle, time_step, True)

				
					fig = plt.gcf()
					fig.savefig(join(params.savedict.savepath, rf"murmurations_{t_now:0>3.4f}.jpg"), \
						bbox_inches='tight',facecolor='None')

					if args.save:
						with h5py.File(savename, 'a') as h5file:
							h5file.create_dataset(f'value/time_{t_now:0>3.3f}', data=value_rolling_np, compression="gzip")

				itr_end.record(); itr_end.synchronize()

				gpu_time_buffer.append(cp.cuda.get_elapsed_time(itr_start, itr_end)/1e3)
				info(f't: {time_step} | GPU time: {gpu_time_buffer[-1]:.4f} | Norm: {np.linalg.norm(y, 2):.2f}')
					
				if len(norm_tracker)>20: # and cp.all(cp.diff(norm_tracker[:-5])<.01):
					logger.info("Terminating the game!")
					break
			end_time = cputime()

			info(f"Avg. local time: {sum(gpu_time_buffer)/len(gpu_time_buffer):.4f} secs")
			info(f"Total Time: {(end_time-start_time):.4f} secs")

		if args.verify:
			x0 = np.array([[1.25, 0, pi]])

			#examine to see if the initial state is in the BRAT
			gexam = copy.deepcopy(flock.grid)
			raise NotImplementedError("Verification of Trajectories is not implemented")

	elif strcmp(args.mode, 'stitch'):
		# load the respective brts for all flocks and start stitching
		base_path = params.savedict.savepath
		murmur_files = sorted([m for m in os.listdir(base_path) if m.startswith('murmurations')])
		murmur_dict = {f"flock_{fname.split(sep='_')[2]}": {'spacing': None, 'time': []} for fname in murmur_files}

		start_time = time.time()
		for f in murmur_files:
			flock_num = f.split(sep='_')[2]
			# load each brt stack
			with h5py.File(join(base_path, f), 'r+') as df:
				value_key = [k for k in df.keys()][0]
				keys = [key for key in df[value_key]]
				if 'spacing' in value_key:
					spacing = np.asarray(df["value/spacing"])
				else:
					gmin = np.asarray([[-1.5, -1.5, -np.pi]]).T
					gmax = np.asarray([[1.5, 1.5, np.pi] ]).T
					grid = createGrid(gmin, gmax, 101, 2)
					spacing = grid.dx.flatten()      
				murmur_dict[f'flock_{flock_num}']['spacing'] = tuple(spacing.tolist())

				idx = 0
				for key in keys[1:]:
					murmur_dict[f'flock_{flock_num}']['time'] += [key]
					murmur_dict[f'flock_{flock_num}'][f'{idx:0>3}'] = np.asarray(df[f"{value_key}/{key}"])
					idx += 1
		end_time = time.time()
		logger.info(f"Time to load saved brats: {(end_time-start_time):.4f} secs.")
	else:
		logger.critical("Option <mode> must be one of <stich or single>")
		os._exit()

if __name__ == '__main__':
	# Do not use python profiler: https://docs.cupy.dev/en/stable/user_guide/performance.html
	from cupyx.profiler import benchmark
	print(benchmark(main, (args,), n_repeat=10))