__all__ = ["Flock"]


__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__credits__  	= "There are None."
__license__ 	= "MIT License"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"

__date__        = "Dec. 21, 2021"
__comment__     = "Two Dubins Vehicle in Relative Coordinates"

import random
import hashlib
import cupy as cp
import numpy as np

from LevelSetPy.InitialConditions.shape_ops import shapeUnion
from .bird import Bird
from LevelSetPy.Grids import *
from LevelSetPy.InitialConditions import *
from LevelSetPy.Utilities.matlab_utils import *

class Graph():
    def __init__(self, n, Grids, vertex_set, edges=None):
        """A graph (an undirected graph that is) that models
        the update equations of agents positions on a state space
        (defined as a grid).

        The graph has a vertex set {1,2,...,n} so defined such that
        (i,j) is one of the graph's edges in case i and j are neighbors.
        This graph changes over time since the relationship between neighbors
        can change.

        Paramters
        =========
            .Grids
            n: number of initial birds (vertices) on this graph.
            .V: vertex_set, a set of vertices {1,2,...,n} that represent the labels
            of birds in a flock. Represent this as a list (see class vertex).
            .E: edges, a set of unordered pairs E = {(i,j): i,j \in V}.
                Edges have no self-loops i.e. i≠j or repeated edges (i.e. elements are distinct).

            Lekan Molu, Dec. 2021.
        """
        self.N = n
        if vertex_set is None:
            self.vertex_set = {f"{i+1}":Bird(Grids[i], 1, 1,\
                    None, random.random(), label=f"{i}") for i in range(n)}
        else:
            self.vertex_set = {f"{i+1}":vertex_set[i] for i in range(n)}

        # edges are updated dynamically during game
        self.edges_set = edges

        # obtain the graph params
        self.reset(self.vertex_set[list(self.vertex_set.keys())[0]].w_e)

    def reset(self, w):
        # graph entities: this from Jadbabaie's paper
        self.Ap = np.zeros((self.N, self.N)) #adjacency matrix
        self.Dp = np.zeros((self.N, self.N)) #diagonal matrix of valencies
        self.θs = np.ones((self.N, 1))*w # agent headings
        self.I  = np.ones((self.N, self.N))
        self.Fp = np.zeros_like(self.Ap) # transition matrix for all the headings in this flock

    def insert_vertex(self, vertex):
        if isinstance(vertex, list):
            assert isinstance(vertex, Bird), "vertex to be inserted must be instance of class Vertex."
            for vertex_single in vertex:
                self.vertex_set[vertex_single.label] = vertex_single.neighbors
        else:
            self.vertex_set[vertex.label] = vertex

    def insert_edge(self, from_vertices, to_vertices):
        if isinstance(from_vertices, list) and isinstance(to_vertices, list):
            for from_vertex, to_vertex in zip(from_vertices, to_vertices):
                self.insert_edge(from_vertex, to_vertex)
            return
        else:
            assert isinstance(from_vertices, Bird), "from_vertex to be inserted must be instance of class Vertex."
            assert isinstance(to_vertices, Bird), "to_vertex to be inserted must be instance of class Vertex."
            from_vertices.update_neighbor(to_vertices)
            self.vertex_set[from_vertices.label] = from_vertices.neighbors
            self.vertex_set[to_vertices.label] = to_vertices.neighbors

    def adjacency_matrix(self, t):
        for i in range(self.Ap.shape[0]):
            for j in range(self.Ap.shape[1]):
                for verts in sorted(self.vertex_set.keys()):
                    if str(j) in self.vertex_set[verts].neighbors:
                        self.Ap[i,j] = 1
        return self.Ap

    def diag_matrix(self):
        "build Dp matrix"
        i=0
        for vertex, egdes in self.vertex_set.items():
            self.Dp[i,i] = self.vertex_set[vertex].valence
        return self.Dp

    def update_headings(self, t):
        return self.adjacency_matrix(t)@self.θs

class Flock(Bird):
    def __init__(self, Grids, vehicles, label=1,
                reach_rad=1.0, avoid_rad=1.0):
        """
            Introduction:
            =============
                A flock of Dubins Vehicles. These are patterned after the
                behavior of starlings which self-organize into local flocking patterns.
                The inspiration for this is the following paper:
                    "Interaction ruling animal collective behavior depends on topological
                    rather than metric distance: Evidence from a field study."
                    ~ Ballerini, Michele, Nicola Cabibbo, Raphael Candelier,
                    Andrea Cavagna, Evaristo Cisbani, Irene Giardina, Vivien Lecomte et al.
                    Proceedings of the national academy of sciences 105, no. 4
                    (2008): 1232-1237.

            Parameters:
            ===========
                .Grids: 2 possible types of Grids exist for resolving vehicular dynamics:
                    .single_grid: an np.meshgrid that homes all these birds
                    .multiple Grids: a collection of possibly intersecting Grids
                        where agents interact.
                .vehicles: Bird Objects in a list.
                .id (int): The id of this flock.
                .reach_rad: The reach radius that defines capture by a pursuer.
                .avoid_rad: The avoid radius that defines the minimum distance between
                agents.
        """
        self.N         = len(vehicles)  # Number of vehicles in this flock
        self.label     = label      # label of this flock
        self.avoid_rad = avoid_rad  # distance between each bird.
        self.reach_rad = reach_rad  # distance between birds and attacker.
        self.vehicles  = vehicles   # # number of birds in the flock
        self.init_random = False

        self.grid = Grids
        """
             Define the anisotropic parameter for this flock.
             This gamma parameter controls the degree of interaction among
             the agents in this flock. Interaction decays with the distance, and
             we can use the anisotropy to get information about the interaction.
             Note that if nc=1 below, then the agents
             exhibit isotropic behavior and the aggregation is non-interacting by and large.
        """
        self.gamma = lambda nc: (1/3)*nc
        self.graph = Graph(self.N, self.grid, self.vehicles, None)

        #update neighbors+headings now based on topological distance
        self._housekeeping()

    def _housekeeping(self):
        """
            Update the neighbors and headings based on topological
            interaction.
        """
        # Update neighbors first
        for i in range(self.N):
            # look to the right and update neighbors
            for j in range(i+1,self.N):
                self._compare_neighbor(self.vehicles[i], self.vehicles[j])

            # look to the left and update neighbors
            for j in range(i-1, -1, -1):
                self._compare_neighbor(self.vehicles[i], self.vehicles[j])

        # recursively update each agent's headings based on neighbors
        for idx, agent in enumerate(self.vehicles):
            self._update_headings(agent, idx)

    def _compare_neighbor(self, agent1, agent2):
        "Check if agent1 is a neighbor of agent2."
        if np.abs(agent1.label - agent2.label) < agent1.neigh_rad:
            agent1.update_neighbor(agent2)

    def _update_headings(self, agent, idx, t=None):
        """
            Update the average heading of this flock.

            Parameters:
            ==========
            agent: This agent as a BirdsSingle object.
            t (optional): Time at which we are updating this agent's dynamics.
        """
        # update heading for this agent
        neighbor_headings = [neigh.w_e for neigh in (agent.neighbors)]

        # this maps headings w/values in [0, 2\pi) to [0, \pi)
        θr = (1/(1+agent.valence))*(agent.w_e + np.sum(neighbor_headings))
        agent.w_e = θr

        # bookkeeing on the graph
        self.graph.θs[idx,:] =  θr

    def hamiltonian(self, t, data, value_derivs, finite_diff_bundle):
        """
            By definition, the Hamiltonian is the total energy stored in
            a system. If we have a team of agents moving along in a state
            space, it would inform us that the total Hamiltonian is a union
            (sum) of the respective Hamiltonian of each agent.

            Parameters
            ==========
            value: Value function at this time step, t
            value_derivs: Spatial derivatives (finite difference) of
                        value function's grid points computed with
                        upwinding.
            finite_diff_bundle: Bundle for finite difference function
                .innerData: Bundle with the following fields:
                    .partialFunc: RHS of the o.d.e of the system under consideration
                        (see function dynamics below for its impl).
                    .hamFunc: Hamiltonian (this function).
                    .dissFunc: artificial dissipation function.
                    .derivFunc: Upwinding scheme (upwindFirstENO2).
                    .innerFunc: terminal Lax Friedrichs integration scheme.
        """
        # do housekeeping: update neighbors and headings
        self._housekeeping()

        # randomly drop one agent from the flock for the pursuer to attack: this aids compactness of the zero levelset
        self.attacked_idx = 0 #np.random.choice(len(self.vehicles))

        # update vehicles not under attack
        vehicles = [x for x in self.vehicles if x is not self.vehicles[self.attacked_idx]]

        # get hamiltonian of non-attcked agents
        unattacked_hams  = []
        for vehicle in vehicles:
            ham_x = vehicle.hamiltonian_abs(t, data, value_derivs, finite_diff_bundle)
            unattacked_hams.append(ham_x.get())
        # unattacked_hams = cp.sum(cp.asarray(unattacked_hams), axis=0)

        # try computing the attack of a pursuer against the targeted agent
        attacked_ham = self.vehicles[self.attacked_idx].hamiltonian(t, data, value_derivs, finite_diff_bundle)

        # sum all the energies of the system
        ham = unattacked_hams + [attacked_ham.get() ]
        ham = shapeUnion(ham)

        return cp.asarray(ham)

    def dissipation(self, t, data, derivMin, derivMax, \
                      schemeData, dim):
        """
            Just add the respective dissipation of all the agent's dissipation
        """
        assert dim>=0 and dim <3, "Dubins vehicle dimension has to between 0 and 2 inclusive."

        # update vehicles not under attack
        vehicles = [x for x in self.vehicles if x is not self.vehicles[self.attacked_idx]]

        # get dissipation of non-attcked agents
        alphas  = []
        for vehicle in vehicles:
            diss_x = vehicle.dissipation_abs(t, data, derivMin, derivMax, \
                      schemeData, dim)
            alphas.append(diss_x)

        attacked_alpha = self.vehicles[self.attacked_idx].dissipation(t, data, derivMin, derivMax, schemeData, dim)
        alphas.append(attacked_alpha)

        alphas = [a_ for a_ in alphas if isnumeric(a_)]+[x.item() for x in alphas if isinstance(x, np.ndarray)]
        # print(alphas) 
        # alphas = np.maximum(alphas, dtype=object)
        alphas = np.max(alphas)
        return cp.asarray(alphas)

    def __eq__(self,other):
        if hash(self)==hash(other):
            return True
        return False

    def __repr__(self):
        parent=f"Flock: {self.label}"
        return parent
