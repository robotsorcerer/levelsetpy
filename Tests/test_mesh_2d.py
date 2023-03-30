__all__ 		= ["levelset_viz", "get_grid"]

__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__license__ 	= "Molux Licence"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"

import argparse
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os, sys
from os.path import abspath, dirname, exists, join
sys.path.append(dirname(dirname(abspath(__file__))))
# sys.path.append('../')
sys.path.append(abspath(join('../..')))

from Grids import createGrid
from InitialConditions import *
from Visualization import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


"""
Test Implicit Functions

Lekan Molu, September 07, 2021
"""

parser = argparse.ArgumentParser(description='2D Plotter for Various Implicit Initial Conditions for the Value Function')
parser.add_argument('--delay', '-dl', type=float, default=3, help='pause time between successive updates of plots' )
args = parser.parse_args()

def levelset_viz(g, ax, fig, mesh, title='', savedict=None, fontdict=None, fc='c', ec='k'):
	"""
		Simultaneously visualize the level sets of a value function
		on a 1X3 chart:
		  	Chart 131: 2D Value function as a surface mesh
			Chart 132: 2D Value function as colored contour levels
			Chart 133: 2D Value zero - set as cyan contour.

		Author: Lekan Molux, October 29, 2021
	"""
	ax[0].plot_surface(g.xs[0], g.xs[1], mesh, rstride=1, cstride=1,
				cmap='viridis', edgecolor=ec, facecolor=fc)
	ax[0].set_title(f'{title}', fontdict=fontdict)
	ax[0].set_xticks([])
	ax[0].set_yticks([])
	ax[0].set_zticks([])

	ax[1].contourf(g.xs[0], g.xs[1], mesh, colors=fc)
	ax[1].set_title(f'Contours', fontdict=fontdict)
	ax[1].set_xticks([])
	ax[1].set_yticks([])
	ax[1].set_zticks([])

	ax[2].contour(g.xs[0], g.xs[1], mesh, levels=0, colors=fc)
	# ax[2].set_xlabel('X', fontdict=fontdict)
	# ax[2].set_ylabel('Y', fontdict=fontdict)
	ax[2].grid('on')
	ax[2].set_title(f'2D Zero level set', fontdict=fontdict)
	ax[2].set_xticks([])
	ax[2].set_yticks([])

	fig.tight_layout()
	if savedict["save"]:
		os.makedirs(savedict["savepath"]) if not os.path.exists(savedict["savepath"]) else None
		plt.savefig(join(savedict["savepath"],savedict["savename"]),
					bbox_inches='tight',facecolor='None')
	# fig.canvas.draw()
	# fig.canvas.flush_events()


def get_grid():

	g2min = -2*np.ones((2, 1),dtype=np.float64)
	g2max = +2*np.ones((2, 1),dtype=np.float64)
	g2N = 51*np.ones((2, 1),dtype=np.int64)
	g2 = createGrid(g2min, g2max, g2N, process=True)

	return g2

def main(savedict):
	# generate signed distance function for cylinder
	center = np.array(([[-.5,.5]]), np.float64).T

	g2 = get_grid()
	# shapes generation
	axis_align, radius=2, 1
	cylinder = shapeCylinder(g2, axis_align, center, radius);
	sphere = shapeSphere(g2, center, radius=1)
	sphere2 = shapeSphere(g2, center=np.array(([-0., 0.])).T, radius=1)
	rect  = shapeRectangleByCorners(g2)
	rect2 = shapeRectangleByCorners(g2, np.array([[ -1.0,  -np.inf,  ]]).T, np.array([[ np.inf, -1.0 ]]).T, )
	rect3 = shapeRectangleByCorners(g2, np.array([[ -1.0,  -0.5,  ]]).T, np.array([[ .5, 1.0 ]]).T)
	rect4 = shapeRectangleByCenter(g2, np.array([[ -1.0,  -0.5,  ]]).T, np.array([[ .5, 1.0 ]]).T)
	# Set Ops
	sphere_union = shapeUnion([sphere, sphere2])
	rect_union = shapeUnion([rect, rect3])
	rect_comp = shapeComplement(rect2)
	sph_rect_diff = shapeDifference(sphere, rect)

	fig = plt.figure(figsize=(16, 9))
	gs = gridspec.GridSpec(1, 3, fig)
	ax = [plt.subplot(gs[i], projection='3d') for i in range(2)] + [plt.subplot(gs[2])]

	savedict["savename"] = "cylinder_2d.jpg"
	levelset_viz(g2, ax, fig, cylinder, title='Cylinder', savedict=savedict, fontdict={'fontsize':22, 'fontweight':'bold'})
	plt.pause(args.delay)

	savedict["savename"] = "sphere_2d.jpg"
	plt.clf()
	ax = [plt.subplot(gs[i], projection='3d') for i in range(2)] + [plt.subplot(gs[2])]

	levelset_viz(g2, ax, fig, sphere, title='Sphere', savedict=savedict, fontdict={'fontsize':22, 'fontweight':'bold'})
	plt.pause(args.delay)

	savedict["savename"]="sphere2_2d.jpg"
	plt.clf()
	ax = [plt.subplot(gs[i], projection='3d') for i in range(2)] + [plt.subplot(gs[2])]

	levelset_viz(g2, ax, fig, sphere2, title='Sphere, C=(-.5, .5)', savedict=savedict, fontdict={'fontsize':22, 'fontweight':'bold'})
	plt.pause(args.delay)

	savedict["savename"]="rect_2d.jpg"
	plt.clf()
	ax = [plt.subplot(gs[i], projection='3d') for i in range(2)] + [plt.subplot(gs[2])]

	levelset_viz(g2, ax, fig, rect, title='Unit Square@Origin', savedict=savedict, fontdict={'fontsize':22, 'fontweight':'bold'})
	plt.pause(args.delay)

	savedict["savename"]="rect2_2d.jpg"
	plt.clf()
	ax = [plt.subplot(gs[i], projection='3d') for i in range(2)] + [plt.subplot(gs[2])]

	levelset_viz(g2, ax, fig, rect2, title='Rect by Corners', savedict=savedict, fontdict={'fontsize':22, 'fontweight':'bold'})
	plt.pause(args.delay)

	savedict["savename"]="rect3_2d.jpg"
	plt.clf()
	ax = [plt.subplot(gs[i], projection='3d') for i in range(2)] + [plt.subplot(gs[2])]

	levelset_viz(g2, ax, fig, rect3, title='RectCorner: [1,-0.5], W: [0.5,1.0]', savedict=savedict, fontdict={'fontsize':22, 'fontweight':'bold'})
	plt.pause(args.delay)

	savedict["savename"]="rect4_2d.jpg"
	plt.clf()
	ax = [plt.subplot(gs[i], projection='3d') for i in range(2)] + [plt.subplot(gs[2])]

	levelset_viz(g2, ax, fig, rect4, title='Rect. Centered', savedict=savedict, fontdict={'fontsize':22, 'fontweight':'bold'})
	plt.pause(args.delay)

	# Show Unions
	savedict["savename"]="sphere_union_2d.jpg"
	plt.clf()
	ax = [plt.subplot(gs[i], projection='3d') for i in range(2)] + [plt.subplot(gs[2])]

	levelset_viz(g2, ax, fig, sphere_union, title='Spheres+Sphere', savedict=savedict, fontdict={'fontsize':22, 'fontweight':'bold'})
	plt.pause(args.delay)

	savedict["savename"]="rect_union_2d.jpg"
	plt.clf()
	ax = [plt.subplot(gs[i], projection='3d') for i in range(2)] + [plt.subplot(gs[2])]

	levelset_viz(g2, ax, fig, rect_union, title='Union of 2 Rects', savedict=savedict, fontdict={'fontsize':22, 'fontweight':'bold'})
	plt.pause(args.delay)

	savedict["savename"]="rect_comp_2d.jpg"
	plt.clf()
	ax = [plt.subplot(gs[i], projection='3d') for i in range(2)] + [plt.subplot(gs[2])]

	levelset_viz(g2, ax, fig, rect_comp, title='Rect Complement', savedict=savedict, fontdict={'fontsize':22, 'fontweight':'bold'})
	plt.pause(args.delay)

	savedict["savename"]="sph_rect_diff_2d.jpg"
	plt.clf()
	ax = [plt.subplot(gs[i], projection='3d') for i in range(2)] + [plt.subplot(gs[2])]

	levelset_viz(g2, ax, fig, sph_rect_diff, title='Sphere-Rect Diff', savedict=savedict, fontdict={'fontsize':22, 'fontweight':'bold'})
	plt.pause(args.delay)




if __name__ == '__main__':
	savedict = dict(save=True, savename='cyl_2d.jpg',\
					savepath=join("/opt/LevPy/Shapes2D"))
	# plt.ion()

	main(savedict)
