
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os, sys
from os.path import abspath, dirname, exists, join
sys.path.append(dirname(dirname(abspath(__file__))))

from Grids import createGrid
from InitialConditions import *
from Visualization import *


"""
Test Implicit Functions

Lekan Molu, September 07, 2021
"""


def get_grid():

    g2min = -2*np.ones((2, 1),dtype=np.float64)
    g2max = +2*np.ones((2, 1),dtype=np.float64)
    g2N = 51*np.ones((2, 1),dtype=np.int64)
    g2 = createGrid(g2min, g2max, g2N, process=True)

    return g2


def main(savedict):
    delay = 1
    block=False
    # generate signed distance function for cylinder
    center = np.array(([[-.5,.5]]), np.float64).T

    g2 = get_grid()
    axis_align, radius=2, 1
    cylinder = shapeCylinder(g2, axis_align, center, radius);
    sphere = shapeSphere(g2, center, radius=1)
    sphere2 = shapeSphere(g2, center=np.array(([-0., 0.])).T, radius=1)
    rect = shapeRectangleByCorners(g2)
    rect2 = shapeRectangleByCorners(g2, np.array([[ -1.0,  -np.inf,  ]]).T, np.array([[ np.inf, -1.0 ]]).T, )
    rect3 = shapeRectangleByCorners(g2, np.array([[ -1.0,  -0.5,  ]]).T, np.array([[ .5, 1.0 ]]).T)
    rect4 = shapeRectangleByCenter(g2, np.array([[ -1.0,  -0.5,  ]]).T, np.array([[ .5, 1.0 ]]).T)
    # Set Ops
    sphere_union = shapeUnion(sphere, sphere2)
    rect_union = shapeUnion(rect, rect3)
    rect_comp = shapeComplement(rect2)
    sph_rect_diff = shapeDifference(sphere, rect)

    savedict["savename"]="cylinder_2d.jpg"
    show2D(g2, cylinder, title='Cylinder', winsize=(12, 7), ec='m', fc=None, savedict=savedict)
    savedict["savename"]="sphere_2d.jpg"
    show2D(g2, sphere, title='Sphere', winsize=(12, 7), ec='m', fc=None, savedict=savedict)
    savedict["savename"]="sphere2_2d.jpg"
    show2D(g2, sphere2, title='Sphere, C=(-.5, .5)', winsize=(12, 7), ec='m', fc=None, savedict=savedict)
    savedict["savename"]="rect_2d.jpg"
    show2D(g2, rect, title='Unit Square Origin', winsize=(12, 7), ec='m', fc=None, savedict=savedict)
    savedict["savename"]="rect2_2d.jpg"
    show2D(g2, rect2, title='-Z&-X of [ -1, -1 ]', winsize=(12, 7), ec='m', fc=None, savedict=savedict)
    savedict["savename"]="rect3_2d.jpg"
    show2D(g2, rect3, title='RectCorner: [1,-0.5], W: [0.5,1.0]', winsize=(12, 7), ec='m', fc=None, savedict=savedict)
    savedict["savename"]="rect4_2d.jpg"
    show2D(g2, rect4, title='RectCent: [1,-0.5], W: [0.5,1.0]', winsize=(12, 7), ec='m', fc=None, savedict=savedict)
    # Show Unions
    savedict["savename"]="sphere_union_2d.jpg"
    show2D(g2, sphere_union, title='Union of 2 Spheres', winsize=(12, 7), ec='m', fc=None, savedict=savedict)
    savedict["savename"]="rect_union_2d.jpg"
    show2D(g2, rect_union, title='Union of 2 Rects', winsize=(12, 7), ec='m', fc=None, savedict=savedict)
    savedict["savename"]="rect_comp_2d.jpg"
    show2D(g2, rect_comp, title='Rect Complement', winsize=(12, 7), ec='m', fc=None, savedict=savedict)
    savedict["savename"]="sph_rect_diff_2d.jpg"
    show2D(g2, sph_rect_diff, title='Sphere-Rect Diff', winsize=(12, 7), ec='m', fc=None, savedict=savedict)



if __name__ == '__main__':
    savedict = dict(save=True, savename='cyl_2d.jpg',\
                    savepath=join("..", "jpeg_dumps"))
    main(savedict)
