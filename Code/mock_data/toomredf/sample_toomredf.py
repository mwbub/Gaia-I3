import sys
sys.path.append('..')

import os
import numpy as np
from toomredf import toomredf

def sample(n, R_range, z_range, phi_range, size=1, use_physical=True):
    filename = 'data/({})({}-{})({}-{})({}-{})({})({}).npy'.format(n, *R_range, 
                *z_range, *phi_range, size, use_physical)
    if os.path.exists(filename):
        return np.load(filename)
    
    df = toomredf(n)
    samples = df.sample_cyl(R_range, z_range, phi_range, size=size, 
                            use_physical=use_physical)
    
    if not os.path.exists('data'):
        os.mkdir('data')
    np.save(filename, samples)
    
    return samples