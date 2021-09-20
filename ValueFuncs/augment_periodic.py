import numpy as np
from BoundaryCondition import addGhostPeriodic
from Utilities import isfield, expand

def augmentPeriodicData(g, data):

    # Dealing with periodicity
    for i  in range(g.dim):
        if isfield(g, 'bdry') and (id(g.bdry[i])==id(addGhostPeriodic)):
            # Grid points
            # print(f'g.vs[{i}]: b4 {g.vs[i].shape}, {g.vs[i][-1].shape}, {g.dx[i]}')
            g.vs[i] = np.concatenate((g.vs[i], expand(g.vs[i][-1] + g.dx[i], 1)), 0)
            # print(f'g.vs[{i}]: aft {g.vs[i].shape}')
            # Input data eg. data = cat(:, data, data(:,:,1))
            # indices = np.arange(g.dim, dtype=np.intp)
            # indices[i] = 0
            # print('indices: ', indices, ' data: ', data.shape)
            # to_app = data[np.ix_(indices)]
            to_app = expand(data[i,...], i)
            # print('to_app ', to_app.shape, ' data b4: ', data.shape)
            data = np.concatenate((data, to_app), i)
            # print(i, ' data aft: ', data.shape)

    return g, data
