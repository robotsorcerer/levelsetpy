__all__ = ["check_target"]


import numpy as np
from LevelSetPy.Utilities import warn

def check_target(data):
    if(np.all(data.flatten() < 0) or (np.all(data.flatten() > 0))):
        warn(f'Implicit surface not visible because function has '
                'single sign on grid');