import numpy as np
import logging

logger = logging.getLOgger(__name__)

class Bundle(object):
    def __init__(self, dicko):
        for var, val in dicko.items():
            object.__setattr__(self, var, val)

def error(arg):
    assert isinstance(arg, str), 'logger.fatal argument must be a string'
    logger.fatal(arg)

def length(A):
    return max(A.shape)

def size(A):
    return A.shape

def numel(A):
    return np.size(A)

def numDims(A):
    return len(A.shape)

def expand(x, ax):
    return np.expand_dims(x, ax)

def ones(x, cols):
    return np.ones((x, cols))

def zeros(rows, cols):
    return np.zeros((rows, cols))

def ndims(x):
    return len(x.shape)

def isvector(x):
    assert numDims(x)>1, 'x must be a 1 x n vector or nX1 vector'
    m,n= x.shape
    if (m==1) or (n==1):
        return True
    else:
        return False

def isColumnLength(x1, x2):
    return ((ndims(x1) == 2) and (x1.shape[0] == x2) and (x1.shape[1] == 1))

def cell(grid_min, dim=1):
    if dim!=1:
        raiseNotImplementedError('This has not been implemented for n>1 cells')
    return [[] for i in range(len(grid_min))]

def iscell(cs):
    if isinstance(cs, list):
        return True
    else:
        return False

def isscalar(x):
    # or simply return len(x)==1
    if not (isinstance(x, np.ndarray) or isinstance(x, list)):
        x = [x]
    return len(x)==1
