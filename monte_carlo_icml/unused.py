import math
import torch 
from torch import nn
from torch.utils.data import DataLoader, Dataset, IterableDataset

class RocketsIterableDataset(Dataset):
    def __init__(self, u_bound=1, w_bound = 1, grid_bound = 64, 
            grid_min = torch.tensor((-grid_bound, -grid_bound, -pi/2)), 
            grid_max = torch.tensor((grid_bound, grid_bound, pi/2)), 
            resolution = 100, g = 32, u_e = 1, 
            u_p = -1, a= 1, u = 1, batch_size=16):
        super(RocketsIterableDataset, self).__init__()
        assert batch_size > 0, "Batch size must be positive"
        self.batch_size = batch_size
        self.u_bound = u_bound
        self.w_bound = w_bound
        self.grid_bound = grid_bound
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.resolution = resolution
        self.g = g
        self.u_e = u_e
        self.u_p = u_p
        self.a = a
        self.u = u

        states = Bundle(dict(x=torch.linspace(-grid_bound, grid_bound, resolution),
                        z=torch.linspace(-grid_bound, grid_bound, resolution),
                        theta=torch.linspace(-torch.pi/2, torch.pi/2, resolution)))
        states.time = torch.linspace(0, 1, resolution)

        system=Bundle(dict())  
        system.states = states
        system.xdot = [
                        a * torch.cos(states.theta) + u_e * states.x,
                        a * torch.sin(states.theta) + a + u_e*states.x- g, 
                        (u_p - u_e)*torch.ones_like(states.x)
                ]
        system.hamfunc = lambda p1, p2, p3: -a * p1 * torch.cos(states.theta) - \
                                p2 * (g - a - a*torch.sin(states.theta)) - \
                                u * abs(p1*torch.sin(states.x) +p3) + \
                                u * abs(p2*states.x + p3)
        
        self.start = 0
        self.end = len(self.resolution)

    def sample(self):
        t_indx = torch.multinomial(states.time, self.batch_size, replacement=False)  
        x = self.states.x[t_indx]
        z = self.states.z[t_indx]
        theta = self.states.theta[t_indx]

    # def __iter__(self):
    #     worker_info = torch.utils.data.get_worker_info()
    #     if worker_info is None:  # single-process data loading, return the full iterator
    #         iter_start = self.start
    #         iter_end = self.end
    #     else:  # in a worker process
    #         # split workload
    #         per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
    #         worker_id = worker_info.id
    #         iter_start = self.start + worker_id * per_worker
    #         iter_end = min(iter_start + per_worker, self.end)

    #         samples = dict()
    #         x = self.states.x[idx]
    #         z = self.states.z[idx]
    #         theta = self.states.theta[idx]
    #         sample_states = torch.meshgrid(*[x, z, theta], indexing='ij')

    #         # samples['states'] = torch.stack([self.states.xs[0][idx], self.states.xs[1][idx], self.states.xs[2][idx]], dim=0)
    #         samples['time'] = self.states.time[idx]

    #         xdot_x = self.system.xdot[0][idx]
    #         xdot_z = self.system.xdot[1][idx]
    #         xdot_theta = self.system.xdot[2][idx]
    #         sample_xdot = torch.meshgrid(*[xdot_x, xdot_z, xdot_theta], indexing='ij')

    #     return iter(range(iter_start, iter_end))


        # make states 3D 
        # states.xs = torch.meshgrid(*[states.x, states.z, states.theta], indexing='ij')


        # self.states = states
        # self.system = system
    
    def __len__(self):
        return len(self.states.x)
    
    def __getitem__(self, idx):
        samples = dict()
        x = self.states.x[idx]
        z = self.states.z[idx]
        theta = self.states.theta[idx]
        sample_states = torch.meshgrid(*[x, z, theta], indexing='ij')

        # samples['states'] = torch.stack([self.states.xs[0][idx], self.states.xs[1][idx], self.states.xs[2][idx]], dim=0)
        samples['time'] = self.states.time[idx]

        xdot_x = self.system.xdot[0][idx]
        xdot_z = self.system.xdot[1][idx]
        xdot_theta = self.system.xdot[2][idx]
        sample_xdot = torch.meshgrid(*[xdot_x, xdot_z, xdot_theta], indexing='ij')

    # samples['xdot'] = torch.stack([sample_xdot[0], sample_xdot[1], sample_xdot[2]], dim=0)
    samples['states']=sample_states
    samples['xdot']=sample_xdot
    # samples.hamfunc = self.system.hamfunc[idx]
    
    # return samples