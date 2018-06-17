import sys
sys.path.append('..')

import numpy as np
from galpy.util.bovy_coords import rect_to_cyl
from search.search_local import search_phase_space
from sample import generate_sample_data

def get_sample_params(u0, v0, w0, epsilon, parallax_cut=True):
    gaia_data = search_phase_space(u0, v0, w0, 0, 0, 0, epsilon, 0, 
                                   parallax_cut=parallax_cut)
    R, phi, z = rect_to_cyl(*gaia_data.T[:3])
    n = len(gaia_data)
    phi_range = [np.min(phi), np.max(phi)]
    r_range = [np.min(R), np.max(R)]
    
    return n, phi_range, r_range