__all__ = [
            "Bundle", "ZEROS_TYPE", "ONES_TYPE", "rad2deg", "deg2rad",
            "realmin", "DEFAULT_ORDER", "mat_like_array", "index_array",
            "quickarray", "ismember", "omin", "omax", "strcmp",
            "isbundle", "isfield", "cputime","error", "realmax", "eps",
            "info","warn", "debug", "length","size",  "to_column_mat", "numel",
            "numDims", "ndims", "expand", "ones", "zeros", "isvector",
            "isColumnLength", "cell",  "iscell", "isnumeric", "isfloat", "isscalar",
]


__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__comment__     = "Common matlab functions."
__credits__  	= "There are None."
__license__ 	= "MIT License"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"


import time
import math
import numbers
import sys, copy
import numpy as np
import logging

logger = logging.getLogger(__name__)

# DEFAULT TYPES
ZEROS_TYPE = np.int64
ONES_TYPE = np.int64
realmin = sys.float_info.min
realmax = sys.float_info.max
eps     = sys.float_info.epsilon
DEFAULT_ORDER = "C"




class Bundle(object):
    def __init__(self, dicko):
        """
            This class creates a Bundle similar to matlab's
            struct class.
        """
        for var, val in dicko.items():
            object.__setattr__(self, var, val)

    def __dtype__(self):
        return Bundle

    def __len__(self):
        return len(self.__dict__.keys())

    def keys(self):
        return list(self.__dict__.keys())

def mat_like_array(start, end, step=1):
    """
        Generate a matlab-like array start:end
        Subtract 1 from start to account for 0-indexing
    """
    return list(range(start-1, end, step))

def index_array(start=1, end=None, step=1):
    """
        Generate an indexing array for nice slicing of
        numpy-like arrays.
        Subtracts 1 from start to account for 0-indexing
        in python.
    """
    assert end is not None, "end in index array must be an integer"
    return np.arange(start-1, end, step, dtype=np.intp)

def quickarray(start, end, step=1):
    "A quick python array."
    return list(range(start, end, step))


def ismember(a, b):
    "Determines if b is a member of the Hash Table a."
    # See https://stackoverflow.com/questions/15864082/python-equivalent-of-matlabs-ismember-function
    bind = {}
    for i, elt in enumerate(b):
        if elt not in bind:
            bind[elt] = i
    return [bind.get(itm, None) for itm in a]  # None can be replaced by any other "not in b" value

def omin(y, ylast):
    "Determines the minimum among both y and ylast arrays."
    if y.shape == ylast.shape:
        temp = np.vstack((y, ylast))
        return np.min(temp)
    else:
        ylast = expand(ylast.flatten(), 1)
        if y.shape[-1]!=1:
            y = expand(y.flatten(), 1)
        temp = np.vstack((y, ylast))
    return np.min(temp)

def omax(y, ylast):
    "Determines the maximum among both y and ylast arrays."
    if y.shape == ylast.shape:
        temp = np.vstack((y, ylast))
        return np.max(temp)
    else:
        ylast = expand(ylast.flatten(), 1)
        if y.shape[-1]!=1:
            y = expand(y.flatten(), 1)
        temp = np.vstack((y, ylast))
    return np.max(temp)

def strcmp(str1, str2):
    "Compares if strings str1 and atr2 are equal."
    if str1==str2:
        return True
    return False

def isbundle(bund):
    "Determines if bund is an instance of the class Bundle."
    if isinstance(bund, Bundle):
        return True
    return False

def isfield(bund, field):
    "Determines if field is an element of the class Bundle."
    return True if field in bund.__dict__.keys() else False

def cputime():
    "Ad-hoc current time function."
    return time.perf_counter()

def error(arg):
    "Pushes std errors out to screen."
    assert isinstance(arg, str), 'logger.fatal argument must be a string'
    raise ValueError(arg)

def info(arg):
    "Pushes std info out to screen."
    assert isinstance(arg, str), 'logger.info argument must be a string'
    logger.info(arg)

def warn(arg):
    "Pushes std watn logs out to screen."
    assert isinstance(arg, str), 'logger.warn argument must be a string'
    logger.warn(arg)

def debug(arg):
    "Pushes std debug logs out to screen."
    assert isinstance(arg, str), 'logger.warn argument must be a string'
    logger.debug(arg)

def length(A):
    "Length of an array A similar to matlab's length function."
    if isinstance(A, list):
        A = np.asarray(A)
    return max(A.shape)

def size(A, dim=None):
    "Size of a matrix A. If dim is specified, returns the size for that dimension."
    if isinstance(A, list):
        A = np.asarray(A)
    if dim is not None:
        return A.shape[dim]
    return A.shape

def to_column_mat(A):
    "Transforms a row-vector array A to a column array."
    n,m = A.shape
    if n<m:
        return A.T
    else:
        return A

def numel(A):
    "Returns the number of elements in an array."
    if isinstance(A, list):
        A = np.asarray(A)
    return np.size(A)

def numDims(A):
    "Returns the numbers of dimensions in an array."
    if isinstance(A, list):
        A = np.asarray(A)
    return A.ndim

def ndims(A):
    "We've got to deprecate this."
    return numDims(A)

def expand(x, ax):
    "Expands an array along axis, ax."
    return np.expand_dims(x, ax)

def ones(rows, cols=None, dtype=ONES_TYPE):
    "Generates a row X col array filled with ones."
    if cols is not None:
        shape = (rows, cols)
    else:
        shape = (rows, rows)
    return np.ones(shape, dtype=dtype)

def zeros(rows, cols=None, dtype=ZEROS_TYPE):
    "Generates a row X col array filled with zeros."
    if cols is not None:
        shape = (rows, cols)
    else:
        if isinstance(rows, tuple):
            shape = rows
        else:
            shape = (rows, rows)
    return np.zeros(shape, dtype=dtype)

def isvector(x):
    "Determines if x is a vector."
    assert numDims(x)>1, 'x must be a 1 x n vector or nX1 vector'
    m,n= x.shape
    if (m==1) or (n==1):
        return True
    else:
        return False

def isColumnLength(x1, x2):
    "Determines if x1 and x2 have the same length along their second dimension."
    if isinstance(x1, list):
        x1 = np.expand_dims(np.asarray(x1), 1)
    return ((ndims(x1) == 2) and (x1.shape[0] == x2) and (x1.shape[1] == 1))

def cell(grid_len, dim=1):
    "Returns a matlab-like cell."
    if dim!=1:
        logger.fatal('This has not been implemented for n>1 cells')
    return [np.nan for i in range(grid_len)]

def iscell(cs):
    "Determines if cs is an instance of a cell."
    if isinstance(cs, list): # or isinstance(cs, np.ndarray):
        return True
    else:
        return False

def isnumeric(A):
    "Determines if A is a numeric type."
    if isinstance(A, numbers.Number):
        return True
    else:
        return False

def isfloat(A):
    "Determines if A is a float type."
    if isinstance(A, np.ndarray):
        dtype = A.dtype
    else:
        dtype = type(A)

    acceptable_types=[np.float64, np.float32, float]

    if dtype in acceptable_types:
        return True
    return False

def isscalar(x):
    "Determines if s is a scalar."
    if (isinstance(x, np.ndarray) and numel(x)==1):
        return True
    elif (isinstance(x, np.ndarray) and numel(x)>1):
        return False
    elif not (isinstance(x, np.ndarray) or isinstance(x, list)):
        return True

def rad2deg(x):
    return (x * 180)/math.pi

def deg2rad(x):
    return x*(math.pi/180)
