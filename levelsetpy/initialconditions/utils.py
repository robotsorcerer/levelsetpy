__all__ = ["check_target"]


__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__credits__  	= "There are None."
__license__ 	= "MIT License"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"


import numpy as np
from levelsetpy.utilities import warn

def check_target(data):
    if(np.all(data.flatten() < 0) or (np.all(data.flatten() > 0))):
        warn(f'Implicit surface not visible because function has '
                'single sign on grid');
