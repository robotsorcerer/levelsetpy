import numpy as np
from BoundaryCondition import addGhostPeriodic

def augmentPeriodicData(g, data):

    # Dealing with periodicity
    for i  in range(g.dim):
        # these might require a closer supervision/debug
        if isfield(g, 'bdry') and isinstance(g.bdry[i], addGhostPeriodic):
            # Grid points
            g.vs[i] = np.concatenate((g.vs[i], g.vs[i][-1] + g.dx[i]), 0);

            # Input data; eg. data = cat(:, data, data(:,:,1))
            colons1 = [[':'] for i in range(g.dim)]
            colons1[i] = 1;
            data = np.concatenate((data, data[colons1[:]]), i)

    return g, data
