__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__license__ 	= "MIT License"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"

import matplotlib.pyplot as plt
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os, sys
from os.path import abspath, dirname, exists, join
sys.path.append(dirname(dirname(abspath(__file__))))

from levelsetpy.grids import createGrid
from levelsetpy.initialconditions import *
from levelsetpy.visualization.mesh_implicit import implicit_mesh

fig = plt.figure(figsize=(16, 9))
def get_grid():

	g3min = -.6*np.ones((3, 1),dtype=np.float64)
	g3max = +6*np.ones((3, 1),dtype=np.float64)
	g3N = 51*np.ones((3, 1),dtype=np.int64)
	g3 = createGrid(g3min, g3max, g3N, process=True)

	return g3

def slender_cylinder(g3):
	axis_align, radius=2, .5
	center = 2*np.ones((3, 1), np.float64)
	cylinder = shapeCylinder(g3, axis_align, center, radius);

	spacing = tuple(g3.dx.flatten().tolist())
	mesh = implicit_mesh(cylinder, level=0., spacing=spacing)

	ax = plt.gca()
	plt.clf()
	ax = fig.add_subplot(121, projection='3d')
	ax.add_collection3d(mesh)

	ax.set_xlabel("x-axis")
	ax.set_ylabel("y-axis")
	ax.set_zlabel("z-axis")

	ax.set_xlim(1.5, 3)
	ax.set_ylim(2.5, 4.0)
	ax.set_zlim(-2, 4.0)

	plt.tight_layout()
	plt.pause(5)

def cylinder_sphere(g3, savedict):
    spacing = tuple(g3.dx.flatten().tolist())

    # generate signed distance function for cylinder
    center = 2*np.ones((3, 1), np.float64)
    ignoreDim, radius=2, 1.5
    cylinder = shapeCylinder(g3, ignoreDim, center, radius);
    cyl_mesh = implicit_mesh(cylinder, level=0., spacing=spacing)

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(121, projection='3d')
    #ax.plot3D(g3.xs[0].flatten(), g3.xs[1].flatten(), g3.xs[2].flatten(), color='cyan')
    ax.add_collection3d(cyl_mesh)

    fontdict = {'fontsize':12, 'fontweight':'bold'}
    ax.set_xlabel("X-axis", fontdict = fontdict)
    ax.set_ylabel("Y-axis", fontdict = fontdict)
    ax.set_zlabel("Z-axis", fontdict = fontdict)
    ax.set_title('Zero Level Set: Cylinder', fontdict = fontdict)

    xlim = (min(cylinder[0].ravel()), max(cylinder[0].ravel()))
    ylim = (min(cylinder[1].ravel()), max(cylinder[1].ravel()))
    zlim = (min(cylinder[2].ravel()), max(cylinder[2].ravel()))

    gxlims = (min(g3.vs[0]), max(g3.vs[0]))
    gylims = (min(g3.vs[1]), max(g3.vs[1]))
    gzlims = (min(g3.vs[2]), max(g3.vs[2]))

    ax.set_xlim(xlim[0]-gxlims[0], xlim[1]-gxlims[1])
    ax.set_ylim(ylim[0]-gylims[0], ylim[1]-gylims[1])
    ax.set_zlim(0, zlim[1])

    sphere = shapeSphere(g3, center, radius=3)
    sphere_mesh = implicit_mesh(sphere, level=0., spacing=spacing, edge_color='k', face_color='g')

    ax2 = fig.add_subplot(122, projection='3d')
    #ax2.plot3D(g3.xs[0].flatten(), g3.xs[1].flatten(), g3.xs[2].flatten(), color='cyan')
    ax2.add_collection3d(sphere_mesh)
    ax2.view_init(elev=30., azim=10.)


    xlim = (min(sphere[0].ravel()), max(sphere[0].ravel()))
    ylim = (min(sphere[1].ravel()), max(sphere[1].ravel()))
    zlim = (min(sphere[2].ravel()), max(sphere[2].ravel()))

    gxlims = (min(g3.vs[0]), max(g3.vs[0]))
    gylims = (min(g3.vs[1]), max(g3.vs[1]))
    gzlims = (min(g3.vs[2]), max(g3.vs[2]))

    ax2.set_xlim(xlim[0]-gxlims[0], xlim[1]-gxlims[1])
    ax2.set_ylim(ylim[0]-gylims[0], ylim[1]-gylims[1])
    ax2.set_zlim(0, zlim[1])

    ax2.set_xlabel("X-axis", fontdict = fontdict)
    ax2.set_ylabel("Y-axis", fontdict = fontdict)
    ax2.set_zlabel("Z-axis", fontdict = fontdict)
    ax2.set_title('Zero Level Set: Sphere', fontdict = fontdict)


    plt.tight_layout()
    plt.show()

    if savedict["save"]:
        fig.savefig(join(savedict["savepath"],savedict["savename"]),
                    bbox_inches='tight',facecolor='None')

def main(savedict):
	"""
		Tests the visuals of an implicit function.

		Here, a cylinder generated  by a signed distance function from
		the nodes of a grid. The cylinder is specified by its radius and its
		distance from the center of the grid. This can be used to represent obstacles
		in a state space from which an agent must avoid.
	"""
	if savedict["save"] and not exists(savedict["savepath"]):
		os.makedirs(savedict["savepath"])
	plt.ion()

	g3 = get_grid()
	savedict["savename"] = "slender_cylinder.jpeg"
	slender_cylinder(g3)
	fig.canvas.draw()
	fig.canvas.flush_events()

	savedict["savename"] = "sphere_cyl.jpeg"
	cylinder_sphere(g3, savedict)
	fig.canvas.draw()
	fig.canvas.flush_events()


	plt.show()

if __name__ == '__main__':
	savedict = dict(save=False, savename='implicit_mesh.jpg',\
					savepath=join("../../", "jpeg_dumps"))
	main(savedict)
