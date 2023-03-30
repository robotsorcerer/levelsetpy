__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__license__ 	= "Molux Licence"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Testing"

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from LevelSetPy.Visualization.mesh_implicit import implicit_mesh

import os, sys
from os.path import abspath, dirname, exists, join
sys.path.append(dirname(dirname(abspath(__file__))))

def show3D(g=None, mesh=None, winsize=(16, 9), title='Zero Level Set', ax=None, disp=1,
            labelsize=18, linewidth=6, fontdict={'fontsize':12, 'fontweight':'bold'},
            savedict=None, ec='k', fc='r', gen_mesh=True, level=0.):

    """
        Visualize the 3D levelset of an implicit function

        Lekan Molu, September 07, 2021
    """
    if not savedict: savedict = {"save": False}

    if gen_mesh:
        spacing = tuple(g.dx.flatten().tolist())
        mesh = implicit_mesh(mesh, level=level, spacing=spacing,  edge_color='k', face_color='r')

    if not ax: # if no gfigure given, gene figure
        fig = plt.figure(figsize=winsize)
        ax = fig.add_subplot(111, projection='3d')

    if np.any(g):
        ax.plot3D(g.xs[0].flatten(), g.xs[1].flatten(), g.xs[2].flatten(), color='cyan')
    if isinstance(mesh, list):
        for m in mesh:
            m = implicit_mesh(m, level=level, spacing=spacing,  edge_color='k', face_color='r')
            ax.add_collection3d(m)
    else:
        ax.add_collection(mesh.mesh)

        xlim = (mesh.verts[:, 0].min(), mesh.verts[:,0].max())
        ylim = (mesh.verts[:, 1].min(), mesh.verts[:,1].max())
        zlim = (mesh.verts[:, 2].min(), mesh.verts[:,2].max())

    ax.set_xlim3d(*xlim)
    ax.set_ylim3d(*ylim)
    ax.set_zlim3d(*zlim)
    
    ax.set_xlabel("X-axis", fontdict = fontdict)
    ax.set_ylabel("Y-axis", fontdict = fontdict)
    ax.set_zlabel("Z-axis", fontdict = fontdict)
    ax.set_title(title, fontdict = fontdict)

    if savedict["save"]:
        fig.savefig(join(savedict["savepath"],savedict["savename"]),
                    bbox_inches='tight',facecolor='None')
    if disp:
        plt.show()

    return ax
