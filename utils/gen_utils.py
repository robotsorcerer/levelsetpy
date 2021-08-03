import numpy as np
import logging
import time

logger = logging.getLogger(__name__)

class Bundle(object):
    def __init__(self, dicko):
        for var, val in dicko.items():
            object.__setattr__(self, var, val)

def isfield(bund, field):
    return True if field in bund.__dict__.keys() else False

def cputime():
    return time.time()

def error(arg):
    assert isinstance(arg, str), 'logger.fatal argument must be a string'
    logger.fatal(arg)

def length(A):
    if isinstance(A, list):
        A = np.asarray(A)
    return max(A.shape)

def size(A, dim=None):
    if isinstance(A, list):
        A = np.asarray(A)
    if dim:
        return A.shape[dim]
    return A.shape

def numel(A):
    if isinstance(A, list):
        A = np.asarray(A)
    return np.size(A)

def numDims(A):
    if isinstance(A, list):
        A = np.asarray(A)
    return len(A.shape)

def expand(x, ax):
    return np.expand_dims(x, ax)

def ones(x, cols):
    return np.ones((x, cols))

def zeros(rows, cols=None):
    if cols:
        shape = (rows, cols)
    else:
        shape = (rows, rows)
    return np.zeros(shape)

def ndims(x):
    return len(size(x))

def isvector(x):
    assert numDims(x)>1, 'x must be a 1 x n vector or nX1 vector'
    m,n= x.shape
    if (m==1) or (n==1):
        return True
    else:
        return False

def isColumnLength(x1, x2):
    if isinstance(x1, list):
        x1 = np.expand_dims(np.asarray(x1), 1)
    return ((ndims(x1) == 2) and (x1.shape[0] == x2) and (x1.shape[1] == 1))

def cell(grid_len, dim=1):
    if dim!=1:
        logger.fatal('This has not been implemented for n>1 cells')
    return [[] for i in range(grid_len)]

def iscell(cs):
    if isinstance(cs, list) or isinstance(cs, np.ndarray):
        return True
    else:
        return False

def isscalar(x):
    if (isinstance(x, np.ndarray) and numel(x)==1):
        return True
    elif (isinstance(x, np.ndarray) and numel(x)>1):
        return False
    elif not (isinstance(x, np.ndarray) or isinstance(x, list)):
        return True
