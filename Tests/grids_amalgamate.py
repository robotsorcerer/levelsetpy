__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__credits__  	= "There are None."
__license__ 	= "Molux Licence"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"


import copy
from absl import flags, app
import sys, os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import numpy as np
from math import pi
from Utilities import expand, zeros, Bundle, ones, error #, FLAGS.order_type
from Grids import createGrid
from ValueFuncs import *
from Visualization import *
from InitialConditions import shapeCylinder
from DynamicalSystems import *
from SpatialDerivative import upwindFirstENO3a
import matplotlib as mpl
mpl.use("Qt5Agg")
import matplotlib.pyplot as


flags.DEFINE_boolean('verbose', False, 'How much debug info to print.')
# General variables now in flags
flags.DEFINE_boolean('visualize', True, 'Show plots?')
flags.DEFINE_float('pause_time', .1, 'Time to puase between updating the next plot')
flags.DEFINE_string('order_type', 'F', 'Use Fortran order or C order for array indexing and memory management.')
flags.DEFINE_boolean('hj_progress', False, 'Display optimization progress')

FLAGS = flags.FLAGS
FLAGS(sys.argv) # we need to explicitly to tell flags library to parse argv before we can access FLAGS.xxx.

import logging

# print('order_type ', FLAGS.order_type)

if FLAGS.verbose:
	logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
else:
	logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

logger = logging.getLogger(__name__)


def main():
    ## Grid 1
	grid_min = expand(np.array((-5, -5, -pi), order=FLAGS.order_type), ax = 1) # Lower corner of computation domain
	grid_max = expand(np.array((5, 5, pi), order=FLAGS.order_type), ax = 1)   # Upper corner of computation domain
	N = 41*ones(3, 1).astype(int) #expand(np.array((41, 41,  41)), ax = 1)        # Number of grid points per dimension
	pdDims = 2              # 3rd dimension is periodic
	g0 = createGrid(grid_min, grid_max, N, pdDims)

	## target set
	data0 = shapeCylinder(g0, 3, zeros(len(N), 1, np.float64), radius=1)
	# show3D(g=g, mesh=data0, savedict={"save":False})
	# also try shapeRectangleByCorners, shapeSphere, etc.


    ## Grid 2
	grid_min = expand(np.array((-5, -5, -pi), order=FLAGS.order_type), ax = 1) # Lower corner of computation domain
	grid_max = expand(np.array((5, 5, pi), order=FLAGS.order_type), ax = 1)   # Upper corner of computation domain
	N = 41*ones(3, 1).astype(int) #expand(np.array((41, 41,  41)), ax = 1)        # Number of grid points per dimension
	pdDims = 2              # 3rd dimension is periodic
	g0 = createGrid(grid_min, grid_max, N, pdDims)

	## target set
	data0 = shapeCylinder(g0, 3, zeros(len(N), 1, np.float64), radius=1)
	# show3D(g=g, mesh=data0, savedict={"save":False})
	# also try shapeRectangleByCorners, shapeSphere, etc.


	## time vector
	t0 = 0
	tMax =  2
	dt = 0.05
	tau = np.arange(t0, tMax+dt, dt) # account for pythonb's 0-indexing
