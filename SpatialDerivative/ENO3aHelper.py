from Utilities import *

def upwindFirstENO3aHelper(grid, data, dim, approx4, stripDD=False):
    """
     upwindFirstENO3aHelper: helper function for upwindFirstENO3a.

       [ dL, dR, DD ] = upwindFirstENO3aHelper(grid, data, dim, approx4, stripDD)

     Helper function to compute the ENO and WENO directional approximations
       to the first derivative using a divided difference table.

     The approximation is constructed by a divided difference table,
       which is more efficient (although a little more complicated)
       than using the direct equations from O&F section 3.4
       (see upwindFirstENO3b for that version).

     Details of this scheme can be found in O&F, section 3.3,
       where this scheme is equivalent to including the Q_1, Q_2 and Q_3
       terms of the ENO approximation.

     parameters:
       grid	Grid structure (see processGrid.m for details).
       data        Data array.
       dim         Which dimension to compute derivative on.
       approx4     Generate two copies of middle approximation using
                     both left/right and right/left traversal of divided
                     difference tree.  The extra copy is placed in the
                     fourth element of derivL and derivR, and is equivalent
                     to the version in the second element of those cell vectors.
       stripDD     Strip the divided difference tables down to their
                     appropriate size, otherwise they will contain entries
                     (at the D1 and D2 levels) that correspond entirely
                     to ghost cells.

       dL          Cell vector containing the 3 or 4 left approximations
                     of the first derivative (each the same size as data).
       dR          Cell vector containing the 3 or 4 right approximations
                     of the first derivative (each the same size as data).
       DD          Cell vector containing the divided difference tables
                     (optional).

    Copyright Lekan Molu, 8/21/2021 Adopted from
            2004 Ian M. Mitchell (mitchell@cs.ubc.ca)
            LevelSets Toolbox. Ian Mitchell, 1/26/04
    """
    #---------------------------------------------------------------------------
    dxInv = np.divide(1, grid.dx[dim])

    # How big is the stencil?
    stencil = 3

    # Add ghost cells.
    gdata = grid.bdry[dim](data, dim, stencil, grid.bdryData[dim])

    #---------------------------------------------------------------------------
    # Create cell array with array indices.
    sizeData = size(gdata)
    indices1 = cell(grid.dim, 1)
    for i in range(grid.dim):
        indices1[i] = list(range(sizeData[i]+1))

    indices2 = indices1

    #---------------------------------------------------------------------------
    # First divided differences (first entry corresponds to D^1_{-3/2}).
    indices1[dim] = list(range(1,size(gdata, dim)+1))
    indices2[dim] = indices1[dim] - 1
    D1 = dxInv@(gdata[indices1.flatten()] - gdata[indices2.flatten()])

    # Second divided differences (first entry corresponds to D^2_{-1}).
    indices1[dim] = list(range(1,size(D1, dim)+1)) #2 : size(D1, dim)
    indices2[dim] = indices1[dim] - 1
    D2 = 0.5 * dxInv@(D1[indices1.flatten()] - D1[indices2.flatten()])

    # Third divided differences (first entry corresponds to D^3_{-1/2}).
    indices1[dim] = list(range(1,size(D2, dim)+1)) #2 : size(D2, dim)
    indices2[dim] = indices1[dim] - 1
    D3 = (1/3) * dxInv@(D2[indices1.flatten()] - D2[indices2.flatten()])

    #---------------------------------------------------------------------------
    # If we want the unstripped divided difference entries, make a copy now.
    DD = Bundle({ 'D1': D1, 'D2': D2, 'D3': D3 })



    # First divided difference array has 2 extra entries at top and bottom
    #   (from stencil width 3), so strip them off.
    # Now first entry corresponds to D^1_{1/2}.
    indices1[dim] = list(range(2, size(D1, dim) - 1))
    D1 = D1[indices1.flatten()]

    # Second divided difference array has an extra entry at top and bottom
    #   (from stencil width 3), so strip them off.
    # Now first entry corresponds to D^2_0.
    indices1[dim] = list(range(1, size(D2, dim)))
    D2 = D2[indices1.flatten()]

    # If we want the stripped divided difference entries, make a copy now.
    DD = Bundle({ 'D1': D1, 'D2': D2, 'D3': D3 })

    #---------------------------------------------------------------------------
    # First order approx is just the first order divided differences.
    #   Make three copies for the three approximations
    #   (or four, if all four possible approximations are desired).
    if(approx4):
        dL = cell(4,1)
        dR = cell(4,1)
    else:
        dL = cell(3,1)
        dR = cell(3,1)


    # Take leftmost grid.N(dim) entries for left approximation.
    indices1[dim] =list(range(size(D1, dim)))
    dL = D1[indices1.flatten()]

    # Take rightmost grid.N(dim) entries for right approximation.
    indices1[dim] = list(range(1, size(D1, dim)+1)) #2 : size(D1, dim)
    dR = D1[indices1.flatten()]

    #---------------------------------------------------------------------------
    # Each copy gets modified by one of the second order terms.
    #   Second order terms are sorted left to right.
    # We'll build the middle approximation by going left then right
    #   So for second order, use the leftward D2 term (indices1).
    # In the four approximation case, we'll do the other direction as well.

    # Coefficients for second order depend only on left or right approximation
    #   (from O&F, depends only on k = i-1 (left) or k = i (right)).
    coeffL = +1 * grid.dx[dim]
    coeffR = -1 * grid.dx[dim]

    indices1[dim] = list(range(size(D2, dim)-1))
    indices2[dim] = list(range(1, size(D2, dim))) #2 : size(D2, dim) - 1
    dL[0] += coeffL@D2[indices1.flatten()]
    dL[1] += coeffL@D2[indices1.flatten()]
    dL[2] += coeffL@D2[indices2.flatten()]
    if(approx4):
        dL[3] += coeffL@D2[indices2.flatten()]

    indices1[dim] += indices1[dim]
    indices2[dim] += indices2[dim]
    dR[0] += coeffR@D2[indices1.flatten()]
    dR[1] += coeffR@D2[indices1.flatten()]
    dR[2] += coeffR@D2[indices2.flatten()]

    if(approx4):
        dR[3] += coeffR * D2[indices2.flatten()]


    #---------------------------------------------------------------------------
    # Each copy gets modified by one of the third order terms.
    #   Third order terms are sorted left to right.
    # We'll build the middle approximation by going left then right.
    #   So for the third order, use the rightward D3 term (indices2).
    # In the four approximation case, we'll do the other direction as well.

    # Coefficients for third order depend on second order term chosen
    #   (from O&F, depends on k* = k-1 (left choice) or k* = k (right choice)).
    # The second L or R refers to whether we went left or right on the D2 term.
    coeffLL = +2 * np.pow(grid.dx[dim], 2)
    coeffLR = -1 * np.pow(grid.dx[dim], 2)
    coeffRL = -1 * np.pow(grid.dx[dim], 2)
    coeffRR = +2 * np.pow(grid.dx[dim], 2)

    indices1[dim] =list(range(size(D3, dim) - 2))
    dL[0] += coeffLL@D3[indices1.flatten()]
    indices1[dim] +=1
    dL[1] += coeffLL@D3[indices1.flatten()]
    if(approx4):
        dL[3] += coeffLR@D3[indices1.flatten()]

    indices1[dim] +=1
    dL[2] += coeffLR@D3[indices1.flatten()]

    indices1[dim] = list(range(1 ,size(D3, dim) - 1))
    dR[0] += coeffRL@D3[indices1.flatten()]
    indices1[dim]+=1
    dR[1] +=coeffRL@D3[indices1.flatten()]
    if(approx4):
        dR[3] += coeffRR@D3[indices1.flatten()]

    indices1[dim] +=1
    dR[2] += coeffRR@D3[indices1.flatten()]

    return dL, dR, DD
