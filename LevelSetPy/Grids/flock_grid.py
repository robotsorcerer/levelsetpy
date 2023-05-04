__all__ = ['flockGrid']

__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__credits__  	= "There are None."
__license__ 	= "Molux Licence"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"


import numpy as np
from .create_grid import createGrid

def flockGrid(grid_mins = [[-1, -1, -np.pi]], grid_maxs = [[1, 1, np.pi]],\
              dx=.2, num_agents=10, N=101):
    """
        Create a grid for a local flock within a Starlings murmuration.

        Parameters:
        ==========
            .grid_mins: A list of lists containing the minimum dimensions of every subgrid for
            every bird in this flock. This list must have as its first entry the min dims of
            the reference bird.

            .grid_maxs: A list of lists containing the maximum dimensions of every subgrid for
            every bird in this flock. This list must have as its first entry the min dims of
            the reference bird.

            .dx: Equidistant spacing between leading flock's grid and every consecutive follower's
            grid.

            num_agents: The total number of agents in this flock.

            .N: Number of points onm each grid.

        Author: Lekan Molux, Dec. 23, 2021
    """
    Grids = [createGrid(np.asarray(grid_mins), np.asarray(grid_maxs), N=N, pdDims=2)]

    for agent in range(1, num_agents):
        agent_grid_min = [(x-dx) for x in grid_mins[agent-1]]
        grid_mins.append(agent_grid_min)

        agent_grid_maxs = [(x-dx) for x in grid_maxs[agent-1]]
        grid_maxs.append(agent_grid_maxs)

        # create Grids
        Grids +=[createGrid(np.asarray(([grid_mins[agent-1]])), \
                            np.asarray(([grid_maxs[agent-1]])), \
                            N=N, pdDims=2)]

    return Grids
