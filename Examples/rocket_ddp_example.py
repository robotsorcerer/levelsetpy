__all__ = ['Two rockets example from Dreyfus']

__author__ 		= "Lekan Molu"
__copyright__ 	= "2022, Large Hamilton-Jacobi Analysis."
__license__ 	= "Molux License"
__comment__ 	= "Dreyfus' Rocket Launch Example with a First-Order Calculus of Variations Approach."
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"

import sys, os
import argparse
import logging

import numpy as np
import matplotlib.pyplot as plt

from os.path import abspath, join, dirname
# sys.path.append(dirname(dirname(abspath(__file__))))
# sys.path.append(abspath(join('..'))) # be sure LevelSetPy is in the same level of your folder structure as largeBRAT

from LevelSetPy.DDPReach import *
from LevelSetPy.Utilities import *


parser = argparse.ArgumentParser(description='Hamilton-Jacobi Analysis')
parser.add_argument('--silent', '-si', action='store_true', help='silent debug print outs' )
parser.add_argument('--visualize', '-vz', action='store_false', help='visualize level sets?' )
parser.add_argument('--init_cond', '-ic', type=str, default='circle', help='visualize level sets?' )
parser.add_argument('--load_brt', '-lb', action='store_false', help='load saved brt?' )
parser.add_argument('--save', '-sv', action='store_true', help='save figures to disk?' )
parser.add_argument('--verify', '-vf', action='store_true', default=True, help='visualize level sets?' )
parser.add_argument('--direction', '-dr',  action='store_true',  help='direction to grow the level sets. Negative by default?' )
parser.add_argument('--pause_time', '-pz', type=float, default=.1, help='pause time between successive updates of plots' )
args = parser.parse_args()

# Turn off pyplot's spurious dumps on screen
logging.getLogger('matplotlib.font_manager').disabled = True
logger = logging.getLogger(__name__)





if __name__ == '__main__':
    if args.silent:
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    try:
        X = np.array(([1, 1, 64, 9.8]))
        trajopt = VarHJIApprox(eta=.5, rho=.99, dx=.1, grid=None, X=X)
        debug('Started trajectory optimization node')
          #trajopt.X
        for xi in X:
            trajopt.backward(X)


    except KeyboardInterrupt:
        warn("Shutting down traj Opt node.")
