import numpy as np
from LevelSetPy.Utilities import isfield, expand
from LevelSetPy.BoundaryCondition import addGhostPeriodic

def augmentPeriodicData(g, data):

    # Dealing with periodicity
    for i  in range(g.dim):
        if isfield(g, 'bdry') and (id(g.bdry[i])==id(addGhostPeriodic)):
            # Grid points
            g.vs[i] = np.concatenate((g.vs[i], expand(g.vs[i][-1] + g.dx[i], 1)), 0)
            indices = [np.arange(data.shape[j], dtype=np.intp) for j in range(data.ndim)]
            indices[i] = [0]
            to_app = data[np.ix_(*indices)]
            data = np.concatenate((data, to_app), i)

    return g, data
