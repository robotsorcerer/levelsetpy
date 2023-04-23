from Utilities import *

def cellMatrixTrace(A):
    """
        A must be an m X n X p matrix/array

        For every n X p matrix in A, we fine the
        trace of each matrix and add them all up
    """
    assert A.shape<3, 'cellMatrixTrace only works on >2D shape arrays'

    traceA = 0
    for k in range(A.shape[0]):
        traceA += np.trace(A[k])

    return traceA
