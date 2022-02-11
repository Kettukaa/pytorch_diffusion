import torch

def slerp(v0, v1, t, DOT_THRESHHOLD=0.9995):
    r"""Spherical interpolation between two tensors
    Arguments:
        v0 (tensor): The first point to be interpolated from. 
        v1 (tensor): The second point to be interpolated from.
        t (float): The ratio between the two points.
        DOT_THRESHHOLD (float): How close should the dot product be to a
                                straight line before deciding to use a linear
                                 interpolation instead.
    Returns:
        Tensor of a single step from the interpolated path between v0 to v1
        at ratio t.  
    """
    v0_copy = torch.clone(v0)
    v1_copy = torch.clone(v1)

    v0 = v0 / torch.norm(v0)
    v1 = v1 / torch.norm(v1)

    dot = torch.sum(v0 * v1)

    if torch.abs(dot) > DOT_THRESHHOLD:
        return torch.lerp(t, v0_copy, c1_copy)
    
    theta_0 = torch.arccos(dot)
    sin_theta_0 = torch.sin(theta_0)

    theta_t = theta_0 * t
    sin_theta_t = torch.sin(theta_t)

    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    v2 = s0 * v0_copy + s1 * v1_copy
    return v2

class Noise:
    def __init__(self, shape, n, seeds=None, interp_func=slerp, state_files=('gs1.pt','gs2.pt'), device='cuda'):
        self.shape = shape
        self.n = n
        self.g1 = torch.Generator(device=device)
        self.g2 = torch.Generator(device=device)
        self.state1 = None
        self.state2 = None
        self.device = device
        self.interp_func = interp_func

        # If no seeds are provided it is implied that a state is being loaded
        if seeds is None:
            print(f'No seed specified, loading states: {state_files}')
            self.load_states(*state_files)

        # Otherwise, make, save, and load a new state file
        else:
            print('Seeds specified, generating state files...')
            print(f'Shape:{self.shape}\t Seeds:{seeds}\t total states per seed:{n}')
            self.make_states(seeds, n, state_files, self.shape)
            self.load_states(*state_files)
        print()

    def interp(self, n, bs, interp_func=slerp):
        x1, x2 = self.randn(n)
        xs = []
        for b in bs:
            xs.append(self.interp_func(x1, x2, b))
            
        return torch.stack(xs)


    def make_states(self, seeds, n, filenames, *args, **kwargs):
        '''
        Creates two state files of n states from seeds

        args:
            seeds: A list of two integer seeds.
            n: The number of states to store per seed.
            filenames: Where the states are saved
            *args **kwargs: Arguments fed directly into torch.randn
        '''

        g = torch.Generator(device=self.device)
        for seed,filename in zip(seeds, filenames):
            g.manual_seed(seed)

            states = g.get_state().unsqueeze(0)
            for i in range(n-1):
                torch.randn(*args, **kwargs, generator=g, device=self.device)
                states = torch.cat((states, g.get_state().unsqueeze(0)), dim=0)
            torch.save(states, filename)
        print(f'States saved as {filenames}')

    def load_states(self, file1, file2):
        '''Loads states from filenames'''
        self.state1 = torch.load(file1)
        self.state2 = torch.load(file2)
        print(f'loaded states {file1} {file2}')

    def set_states(self, n):
        '''Sets the state of both generators to the nth respective state'''
        self.g1.set_state(self.state1[n].cpu()) # For whatever reason, these need to be on the cpu to work.
        self.g2.set_state(self.state2[n].cpu())

    def randn(self, n):
        '''Returns a two element list containing both noise tensors at the nth state'''
        self.set_states(n)
        return torch.stack([torch.randn(self.shape, generator=g, device=self.device) for g in [self.g1, self.g2]])
