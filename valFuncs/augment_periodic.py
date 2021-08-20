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
            to_app = np.expand_dims(data[..., i], i)
            data = np.concatenate((data, to_app), i)

    return g, data
