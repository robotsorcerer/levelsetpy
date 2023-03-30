
import numpy as np
from LevelSetPy.Utilities import zeros, cell

dubins_default_params = dict(
                            wRange=[-1, 1],
                            speed=5,
                            dRange=zeros(3,2),
                            dims=np.arange(3),
                            end=None,
                            nu = 1,
                            nd = 3,
                            x = None,
                            nx=3,
                            T = 0,
                            u = None,
                            xhist=None,
                            uhist=None,
                            pdim=None,
                            vdim=None,
                            hdim=None,
                            hpxpy=None,
                            hpxpyhist=None,
                            hvxvy=None,
                            hvxvyhist=None,
                            hpv = cell(2,1),
                            hpvhist = cell(2,1),
                            data = None
                            )
