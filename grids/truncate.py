import copy
from Utilities import *
from .process_grid import processGrid

def truncateGrid(gOld, dataOld=None, xmin=None, xmax=None, process=True):
    """
     [gNew, dataNew] = truncateGrid(gOld, dataOld, xmin, xmax)
        Truncates dataOld to be within the bounds xmin and xmax

     Inputs:
       gOld, gNew - old and new grid structures
       dataOld    - data corresponding to old grid structure
       process    - specifies whether to call processGrid to generate
                    grid points

     Output: dataNew    - equivalent data corresponding to new grid structure

     Mo Chen, 2015-08-27
     Lekan Molux, Aug 21, 2021

     Gather indices of new grid vectors that are within the bounds of the old
     grid
    """
    gNew = Bundle(dict(dim=gOld.dim, vs=cell(gOld.dim, 1),
                        N = np.zeros((gOld.dim, 1), dtype=np.int64),
                        min=np.zeros((gOld.dim, 1),  dtype=np.float64),
                        max=np.zeros((gOld.dim, 1),  dtype=np.float64),
                        bdry = gOld.bdry
                    ))
    small = 1e-3

    for i in range(gNew.dim):
        idx = np.logical_and(gOld.vs[i] > xmin[i], gOld.vs[i] < xmax[i])
        gNew.vs[i] = expand(gOld.vs[i][idx], 1)
        gNew.N[i]= len(gNew.vs[i])
        gNew.min[i] = np.min(gNew.vs[i])

        if gNew.N[i] == 1:
            gNew.max[i] = np.max(gNew.vs[i]) + small
        else:
            gNew.max[i] = np.max(gNew.vs[i])
        if gNew.N[i] < gOld.N[i]:
            gNew.bdry[i] = addGhostExtrapolate
    if process:
        gNew = processGrid(gNew)

    if not np.any(dataOld):
        return gNew

    dataNew = []
    # Truncate everything that's outside of xmin and xmax
    if gOld.dim==1:
        if dataOld:
            dataNew = dataOld[np.logical_and(gOld.vs[0]>xmin, gOld.vs[0]<xmax)]


    elif gOld.dim==2:
        if np.any(dataOld):
            r = np.logical_and(gOld.vs[0]>xmin[0], gOld.vs[0]<xmax[0]).squeeze()
            c = np.logical_and(gOld.vs[1]>xmin[1], gOld.vs[1]<xmax[1]).squeeze()
            dataNew = dataOld[r,:][:,c]

    elif gOld.dim==3:
        if np.any(dataOld):
            row_idx = np.logical_and(gOld.vs[0]>xmin[0], gOld.vs[0]<xmax[0]).squeeze()
            col_idx = np.logical_and(gOld.vs[1]>xmin[1], gOld.vs[1]<xmax[1]).squeeze()
            z_idx   = np.logical_and(gOld.vs[2]>xmin[2], gOld.vs[2]<xmax[2]).squeeze()
            dataNew = dataOld[row_idx, :, :][:,col_idx,:][:,:,z_idx]

    elif gOld.dim==4:
        if np.any(dataOld):
            row_idx = np.logical_and(gOld.vs[0]>xmin[0], gOld.vs[0]<xmax[0]).squeeze()
            col_idx = np.logical_and(gOld.vs[1]>xmin[1], gOld.vs[1]<xmax[1]).squeeze()
            z_idx   = np.logical_and(gOld.vs[2]>xmin[2], gOld.vs[2]<xmax[2]).squeeze()
            forth_idx = np.logical_and(gOld.vs[3]>xmin[3], gOld.vs[3]<xmax[3]).squeeze()

            dataNew = dataOld[row_idx, :, :,:][:,col_idx,:,:][:,:,z_idx,:][:,:,:,forth_idx]

    else:
        error('truncateGrid has only been implemented up to 4 dimensions!')

    return gNew, dataNew
