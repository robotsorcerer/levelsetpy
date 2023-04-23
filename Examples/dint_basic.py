#!/usr/bin/env python
# coding: utf-8


import sys, os
import logging
import argparse
import copy, time
import numpy as np
import scipy.linalg as la
from os.path import abspath, join
import matplotlib.pyplot as plt

from os.path import abspath, join, expanduser

# Pull Mister Molux's Libraries
from LevelSetPy.Utilities import *
from LevelSetPy.Grids import createGrid
from LevelSetPy.Visualization import *
from LevelSetPy.DynamicalSystems import DoubleIntegrator

# Value co-state and Lax-Friedrichs upwinding schemes
from LevelSetPy.InitialConditions import *
from LevelSetPy.SpatialDerivative import *
from LevelSetPy.ExplicitIntegration import * 


"""
### Create state space and value function.
 
 + This would be the grid upon which we are going to control 
 the car/unicycle etc -- basically anything that may have 
 double integrator dynamics.
 
 + In addition, we'll generate the analytical solution to the time to reach the origin problem on the grid. Let's call this variable `attr`. 
 
 + Our initial value function will be constructed as an implicit function on the grid with a signed distance function in the geometry of a sphere of radius, $r$.
"""

parser = argparse.ArgumentParser(description='Double Integrator Analysis')
parser.add_argument('--silent', '-si', action='store_true', help='silent debug print outs' )
parser.add_argument('--visualize', '-vz', action='store_false', help='visualize level sets?' )
parser.add_argument('--init_cond', '-ic', type=str, default='ellipsoid', help='visualize level sets?' )
parser.add_argument('--show_levels', '-sl', action='store_true', default=False, help='load saved brt?' )
parser.add_argument('--load_brt', '-lb', action='store_false', help='load saved brt?' )
parser.add_argument('--save', '-sv', action='store_true', help='save figures to disk?' )
parser.add_argument('--verify', '-vf', action='store_true', default=True, help='visualize level sets?' )
parser.add_argument('--direction', '-dr',  action='store_true',  help='direction to grow the level sets. Negative by defalt?' )
parser.add_argument('--pause_time', '-pz', type=float, default=.1, help='pause time between successive updates of plots' )
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

dint = DoubleIntegrator
u_bound = 1
fontdict = {'fontsize':16, 'fontweight':'bold'}

def preprocessing():
    global dint, u_bound

    gmin = np.array(([[-1, -1]]),dtype=np.float64).T
    gmax = np.array(([[1, 1]]),dtype=np.float64).T
    g = createGrid(gmin, gmax, 101, None)

    eps_targ = 1.0
    target_rad = .2

    dint = DoubleIntegrator(g, u_bound)
    attr = dint.mttr() - target_rad
    if strcmp(args.init_cond, 'attr'):
        value_func_init = attr
    elif strcmp(args.init_cond, 'circle'):
        value_func_init = shapeSphere(g, zeros(g.dim, 1), target_rad)
    elif strcmp(args.init_cond, 'square'):
        value_func_init = shapeRectangleByCenter(g, zeros(g.dim, 1),  2*target_rad*ones(g.dim,1))
    elif strcmp(args.init_cond, 'ellipsoid'):
        value_func_init = shapeEllipsoid(g, zeros(g.dim, 1), 1)

    attr = np.maximum(0, attr)

    return g, attr, value_func_init


def show_init_levels(g, attr, value_func_init, savename='ttr_0.jpeg'):

    f, (ax1, ax2) = plt.subplots(1,2,figsize=(25, 9))

    def do_ax(ax, val_func, title='Analytical TTR', color='m'):        
        ax.contour(g.xs[0], g.xs[1], val_func, colors=color, linewidths=[4.5, 4.8, 5.2, 6.5, 7.8], corner_mask=False, linestyles="dashdot")
        ax.set_title(title, fontdict={'fontsize':38, 'fontweight':'bold'})
        ax.set_xlabel(r"$x_1 (m)$", fontdict={'fontsize':30, 'fontweight':'bold'})
        ax.set_xlim([-1.02, 1.02])
        ax.set_ylim([-1.01, 1.01])
        
        ax.set_xticks([-0.75, 0.0, 1.01])
        ax.set_yticks([-0.75, 0.0, 1.01])
        ax.grid()

    do_ax(ax1, attr)
    do_ax(ax2, value_func_init, title='Initial (Approx) TTR', color='b')
    ax1.set_ylabel(r"$x_2 (m/s)$", fontdict={'fontsize':35, 'fontweight':'bold'})

    f.suptitle(f"Stacked levelsets", **fontdict)

    plt.tight_layout()
    if savename:
        savepath = join(base_path, "Dint")
        os.makedirs(savepath) if not os.path.exists(savepath) else None
        f.savefig(join(savepath, savename), bbox_inches='tight',facecolor='None')

    f.canvas.draw()
    f.canvas.flush_events()
    time.sleep(args.pause_time)


dint = DoubleIntegrator
u_bound = 1
fontdict={'fontsize':42, 'fontweight':'bold'}
g, attr, value_func_init = preprocessing()
base_path = "/opt/LevPy"
os.makedirs(base_path) if not os.path.exists(base_path) else None
show_init_levels(g, attr, value_func_init, savename='dint_0.jpeg')


# #### Trajectories in the $x_1$-$x_2$ plane
# 
# \begin{align}
# x_1 &= \xi_1 + \frac{1}{2} \Delta x_2^2 - \frac{1}{2} \Delta \xi_2^2, \text{ where } \quad t = \Delta(x_2 - \xi_2)  \\
# x_2 &= \xi_2 + \Delta t
# \end{align}
# 
# where $\Delta= u \equiv \pm 1$, 
# 
# $t$ is the time it takes to arrive at our goal; 
# 
# and $x_1$ and $x_2$ are the position and velocity of the body respectively.

fontdict = {'fontsize':28, 'fontweight':'bold'}
def show_trajectories(g, attr, x_i):
	### pick a bunch of initial conditions
	Δ = lambda u: u  # u is either +1 or -1
	
	# how much timesteps to consider
	t = np.linspace(-1, 1, 100)
	# do implicit euler integration to obtain x1 and x2
	x1p = np.empty((len(x_i), g.xs[0].shape[1], len(t)))
	x2p = np.empty((len(x_i), g.xs[1].shape[1], len(t)))
	# states under negative control law
	x1m  = np.empty((len(x_i), g.xs[0].shape[1], len(t)))
	x2m = np.empty((len(x_i), g.xs[1].shape[1], len(t)))

	fig, ax1 = plt.subplots(1, 1, figsize=(25,14))

	for i in range(len(x_i)):
		for k in range(len(t)):
			# state trajos under +ve control law
			x2p[i, :,k] = x_i[i][1] + Δ(u_bound) * t[k]
			x1p[i, :,k] = x_i[i][0] + .5 * Δ(u_bound) * x2p[i,:,k]**2 - .5 * Δ(u_bound) * x_i[i][1]**2
			# state trajos under -ve control law
			x2m[i, :,k] = x_i[i][1] + Δ(-u_bound) * t[k]
			x1m[i, :,k] = x_i[i][0]+.3 + .5 * Δ(-u_bound) * x2p[i,:,k]**2 - .5 * Δ(-u_bound) * x_i[i][1]**2


	# Plot a few snapshots for different initial conditions.
	color = iter(plt.cm.inferno_r(np.linspace(.25, 1, 2*len(x_i))))
	# repeat for legends
	for init_cond in range(0, len(x_i)):
		# state trajectories are unique for every initial cond
		# here, we pick the last state
		ax1.plot(x1p[init_cond, -1, :], x2p[init_cond, -1, :], linewidth=4, color=next(color), 			label=rf"x$_{init_cond+1}^+={x_i[init_cond]}$")
		ax1.plot(x1m[init_cond, -1, :], x2m[init_cond, -1, :], '-.', linewidth=4, color=next(color) , 			label=rf"x$_{init_cond+1}^-={x_i[init_cond]}$")

		#plot the quivers
		ax1.grid('on')
		up, vp = x2p[init_cond, -1, ::len(t)//3], [Δ(u_bound)]*len(x2p[init_cond, -1, ::len(t)//3])
		um, vm = x2m[init_cond, -1, ::len(t)//3], [Δ(-u_bound)]*len(x2m[init_cond, -1, ::len(t)//3])
		ax1.quiver(x1p[init_cond, -1, ::len(t)//3], x2p[init_cond, -1, ::len(t)//3], up, vp, angles='xy', width=.0035)
		ax1.quiver(x1m[init_cond, -1, ::len(t)//3], x2m[init_cond, -1, ::len(t)//3], um, vm, angles='xy', width=.0035)

	ax1.set_ylabel(rf"${{x}}_2$", fontdict={'fontsize':43, 'fontweight':'bold'})
	ax1.set_xlabel(rf"${{x}}_1$", fontdict={'fontsize':43, 'fontweight':'bold'})
	ax1.tick_params(axis='both', which='major', labelsize=28)
	ax1.tick_params(axis='both', which='minor', labelsize=28)
	
	ax1.set_yticks([-0.8, 0.0, 0.8])
	ax1.set_xticks([-1.2, 0.0, 1.2])
	
	ax1.set_ylim([-.85, 0.85])
	ax1.set_xlim([-1.0, 1.0])

	ax1.set_title(rf"State Trajectories", fontdict={'fontsize':43, 'fontweight':'bold'})
	# ax1.legend(loc="center left", fontsize=14) 
	ax1.set_facecolor("lavender")
	
	plt.tight_layout()
	savepath = join(base_path, "Dint")
	os.makedirs(savepath) if not os.path.exists(savepath) else None
	fig.savefig(join(savepath, "doub_int_trajos.jpg"), bbox_inches='tight',facecolor='None')
	
	time.sleep(args.pause_time)

xis = [(1,0), (.75, 0), (.5, 0), (.25,0.), (0,0), (-.25,0), (-.5, 0), (-.75, 0), (-1,0)]
show_trajectories(g, attr, xis)


### Plot the switching curve
fontdict = {'fontsize':18, 'fontweight':'bold'}
def show_switch_curve():
	# Plot all vectograms(snapshots) in space and time.
	fig3, ax3 = plt.subplots(1, 1, figsize=(15,18))
	ax3.grid('on')
	color = iter(plt.cm.ocean(np.linspace(.25, 1, 4)))
	ax3.set_facecolor("bisque")
	ax3.plot(dint.Gamma[0,:51], linewidth=3.5, linestyle="dashdot", color='green', label=rf"$\gamma_-$")
	ax3.plot(np.arange(51, 101), dint.Gamma[0,51:], linewidth=3.5, linestyle="dashdot", color='darkviolet', label=rf"$\gamma_+$")
	# ax3.plot(dint.Gamma[0,:], linewidth=3.5, linestyle="dashdot", color='lightpink', label=rf"$\gamma_+$")
	xmin, xmax = ax3.get_xlim()
	ymin, ymax = ax3.get_ylim()
	ax3.hlines(0, xmin, xmax, colors='black', linestyles='solid', label='')
	ax3.vlines(len(dint.Gamma)//2, ymin, ymax, colors='black', linestyles='solid', label='')

	ax3.set_xlim(0, 100)
	ax3.set_ylim(-.5, .5)

	ax3.set_xlabel(rf"${{x}}_1$", fontdict=fontdict)
	ax3.set_ylabel(rf"${{x}}_2$", fontdict=fontdict)
	ax3.set_title(rf"Switching Curve, $\gamma$", fontdict= {'fontsize':10, 'fontweight':'bold'})
	ax3.tick_params(axis='both', which='major', labelsize=10)
	ax3.tick_params(axis='both', which='minor', labelsize=10)
	ax3.legend(fontsize=18)			
	plt.tight_layout()

	savepath = join(base_path, "Dint")
	os.makedirs(savepath) if not os.path.exists(savepath) else None
	fig3.savefig(join(savepath, "switching_curve.jpg"), bbox_inches='tight',facecolor='None')
show_switch_curve()


### Plot the Analytical Time to Reach the origin
def show_attr():
    # Plot all vectograms(snapshots) in space and time.
    fig2, ax2 = plt.subplots(1, 1, figsize=(25,9))
    cdata = ax2.pcolormesh(g.xs[0], g.xs[1], attr, shading="nearest", cmap="magma_r")
    plt.colorbar(cdata, ax=ax2, extend="both")
    ax2.set_xlabel(rf"${{x}}_1$", fontdict=fontdict)
    ax2.set_ylabel(rf"${{x}}_2$", fontdict=fontdict)
    ax2.set_title(r"Reach Time", fontdict={'fontsize':42, 'fontweight': 'bold'})
    ax2.tick_params(axis='both', which='major', labelsize=28)
    ax2.tick_params(axis='both', which='minor', labelsize=18)

    ax2.set_yticks([-0.8, 0.0, 0.8])
    ax2.set_xticks([-0.8, -0.4, 0, 0.4, 0.8])
	
    ax2.set_ylim([-.9, 0.9])
    ax2.set_xlim([-.90, 0.9])
    
    savepath = join(base_path, "Dint")
    os.makedirs(savepath) if not os.path.exists(savepath) else None
    fig2.savefig(join(savepath, "attr.jpg"), bbox_inches='tight',facecolor='None')


# 
# ### Examining the Level Sets
# 
# The analytical time-to-reach is resolved as
# 
# \begin{align}
#   &\left[\left(x_2 + \sqrt{4 x_1 + 2 x_2^2}\right) \, \cdot \, \left(x_1 > \Gamma\right)\right] + 
#             &\left[\left(-x_2 + \sqrt{2x_2^2 - 4 x_1}\right)  \, \cdot \,  (x_1 < \Gamma) \right] +
#             |x_2|  \, \cdot \,  \left(x_1 = \Gamma\right).
# \end{align}
# 
# where $\Gamma$ is the switching curve defined as 
# 
# \begin{align}
#     \Gamma = -\dfrac{1}{2} \cdot x_2 \dot |x_2|
# \end{align}


show_attr()


def isochoner(tstar = 0.5):
    above_curve = (-.5 * dint.grid.xs[1]**2 + .25*  (tstar - dint.grid.xs[1])**2 )*(dint.grid.xs[0]>dint.Gamma)
    below_curve = (.5 * dint.grid.xs[1]**2 - .25 * (tstar +  dint.grid.xs[1])**2 )*(dint.grid.xs[0]<dint.Gamma)
    on_curve =  (-.5 * (dint.grid.xs[1] * tstar))*(dint.grid.xs[0]==dint.Gamma)

    return above_curve, below_curve, on_curve

tt = dint.mttr()
above_curve, below_curve, on_curve = isochoner(tt)
all_isochrones = above_curve+below_curve+on_curve


def isochrones(g, curve, cmap_color, title="Isochrones above switching curve", savename="isochoner_above.jpg"): # Plot all vectograms(snapshots) in space and time.
    fig2, ax2 = plt.subplots(1, 1, figsize=(25,13))
    cdata = ax2.pcolormesh(g.xs[0], g.xs[1], curve, shading="nearest", cmap=cmap_color)
    plt.colorbar(cdata, ax=ax2, extend="both")
    ax2.set_xlabel(rf"${{x}}_1$", fontdict=fontdict)
    ax2.set_ylabel(rf"${{x}}_2$", fontdict=fontdict)
    ax2.set_title(rf"{title}", fontdict={'fontsize':42, 'fontweight': 'bold'})
    ax2.tick_params(axis='both', which='major', labelsize=28)
    ax2.tick_params(axis='both', which='minor', labelsize=18)

    ax2.set_yticks([-0.8, 0.0, 0.8])
    ax2.set_xticks([-0.8, -0.4, 0, 0.4, 0.8])
	
    ax2.set_ylim([-.9, 0.9])
    ax2.set_xlim([-.90, 0.9])

    savepath = join(base_path, "Dint")
    os.makedirs(savepath) if not os.path.exists(savepath) else None
    fig2.savefig(join(savepath, savename), bbox_inches='tight',facecolor='None')

isochrones(g, above_curve, "YlOrBr_r", title="Isochrones above switching curve", savename="isochoner_above.jpg")
isochrones(g, below_curve, "afmhot_r", title="Isochrones below switching curve", savename="isochoner_below.jpg")
isochrones(g, all_isochrones, "inferno_r", title="All isochrones", savename="isochoner_all.jpg")


# Optimize the trajectorie
finite_diff_data = Bundle(dict(innerFunc = termLaxFriedrichs,
            innerData = Bundle({'grid':g,
                'hamFunc': dint.hamiltonian,
                'partialFunc': dint.dissipation,
                'dissFunc': artificialDissipationGLF,
                'CoStateCalc': upwindFirstENO2,
                }),
                positive = False,  # direction to grow the updated level set
            ))

small = 100*eps
t_span = np.linspace(0, 2.0, 20)
options = Bundle(dict(factorCFL=0.75, stats='on', maxStep=realmax,singleStep='off', postTimestep=postTimestepTTR))

y = copy.copy(value_func_init.flatten())
y, finite_diff_data = postTimestepTTR(0, y, finite_diff_data)
value_func = np.asarray(copy.copy(y.reshape(g.shape)))


# Visualization paramters
spacing = tuple(g.dx.flatten().tolist())
params = Bundle({'grid': g,
					'disp': True,
					'labelsize': 16,
					'labels': "Initial 0-LevelSet",
					'linewidth': 2,
					'mesh': value_func,
					'init_conditions': False,
					'pause_time': args.pause_time,
					'level': 0, # which level set to visualize
					'winsize': (25,9),
					'fontdict': Bundle({'fontsize':40, 'fontweight':'bold'}),
					"savedict": Bundle({"save": True,
								"savename": "dint",
								"savepath": "/opt/LevPy/Dint/"})
					})

if args.visualize:
	viz = DoubleIntegratorVisualizer(params)

value_func_all = np.zeros((len(t_span),)+value_func.shape)

cur_time, max_time = 0, t_span[-1]
step_time = (t_span[-1]-t_span[0])/8.0

start_time = time.perf_counter()

idx = 0
cpu_time_buffer = []
while max_time-cur_time > small * max_time:
	cpu_start = time.perf_counter()

	time_step = f"{cur_time:.2f}/{max_time:.2f}"

	y0 = value_func.flatten()
	yoa = attr.flatten()

	t_span = np.hstack([cur_time, min(max_time, cur_time + step_time)])

	# advance one step of integration
	t, y, finite_diff_data = odeCFL2(termRestrictUpdate, t_span, y0, options, finite_diff_data)
	cur_time = t if np.isscalar(t) else t[-1]

	value_func = np.reshape(y, g.shape)

	if args.visualize:
		ls_mesh = value_func
		# store this brt
		value_func_all[idx] = ls_mesh
		viz.update_tube(attr, ls_mesh, cur_time, delete_last_plot=True)
		idx += 1

	cpu_end = time.perf_counter()

	cpu_time_buffer.append(cpu_end-cpu_start)
	
	print(f't: {time_step} | CPU Time: {(cpu_end-cpu_start):.2f}')


end_time = time.perf_counter()

print()
info(f"Avg. local time: {sum(cpu_time_buffer)/len(cpu_time_buffer):.4f} secs")
print(f'Total BRS/BRT execution time {(end_time - start_time):.4f} seconds.')


def interm_levels(g, value_func, title='Closed-Form TTR', color='m', savename='ttr_0.jpeg'):

    f, ax = plt.subplots(1,1,figsize=(25, 9))
     
    ax.contour(g.xs[0], g.xs[1], value_func, levels=8, colors=color, linestyles="solid", linewidths=[4.5, 4.8, 5.2, 6.5, 7.8])
    ax.set_xlabel(r"$x_1 (m)$", fontdict={'fontsize':30, 'fontweight':'bold'})
    ax.set_xlim([-1.02, 1.02])
    ax.set_ylim([-1.01, 1.01])
    
    ax.set_xticks([-0.75, 0.0, 1.01], fontdict={'fontsize':25, 'fontweight':'bold'})
    ax.set_yticks([-0.75, 0.0, 1.01], fontdict={'fontsize':25, 'fontweight':'bold'})
    ax.grid()

    ax.set_ylabel(r"$x_2 (m/s)$", fontdict={'fontsize':35, 'fontweight':'bold'})

    f.suptitle(f"Stacked overapproximated levelsets", **{'fontsize':38, 'fontweight':'bold'})

    plt.tight_layout()
    if savename:
        savepath = join(base_path, "Dint")
        os.makedirs(savepath) if not os.path.exists(savepath) else None
        f.savefig(join(savepath, savename), bbox_inches='tight',facecolor='None')

    plt.show()


if args.show_levels:
	color = iter(plt.cm.inferno_r(np.linspace(.25, 1.25, 8)))
	idx=1; interm_levels(g, value_func_all[idx], color="darkorange", title=f"Stacked overapproximated levelsets", savename=f'dint_{idx+1:0>2}.jpeg')
	idx=2; interm_levels(g, value_func_all[idx], color="darkred", title=f"Stacked overapproximated levelsets", savename=f'dint_{idx+1:0>2}.jpeg')
	idx=3; interm_levels(g, value_func_all[idx], color="darkmagenta", title=f"Stacked overapproximated levelsets", savename=f'dint_{idx+1:0>2}.jpeg')
	idx=4; interm_levels(g, value_func_all[idx], color="orangered", title=f"Stacked overapproximated levelsets", savename=f'dint_{idx+1:0>2}.jpeg')
	idx=5; interm_levels(g, value_func_all[idx], color="deeppink", title=f"Stacked overapproximated levelsets", savename=f'dint_{idx+1:0>2}.jpeg')




