import sys
sys.path.append('..')
sys.path.append('../..')

import os
import dill
import numpy as np
from toomredf import toomredf
from search import search_local
from tools.tools import cyl_to_rect, rect_to_cyl

file = '../../selection/parallax selection with galactic plane/selection_function'
dill_file = open(file, 'rb')
selection = dill.load(dill_file)

def sample(n, R_range, z_range, phi_range, size=1, use_physical=True):
    filename = 'data/({})({}-{})({}-{})({}-{})({})({}).npy'.format(n, *R_range, 
                *z_range, *phi_range, size, use_physical)
    if os.path.exists(filename):
        return np.load(filename)
    
    df = toomredf(n)
    samples = df.sample_cyl(R_range, z_range, phi_range, size=size, 
                            use_physical=use_physical)
    samples = cyl_to_rect(*samples.T)
    
    if not os.path.exists('data'):
        os.mkdir('data')
    np.save(filename, samples)
    
    return samples

def sample_selection(n, R_range, z_range, phi_range, selection, size=1, 
                     use_physical=True, dd=True):
    filename = 'data/(selection)({})({}-{})({}-{})({}-{})({})({})'.format(n, 
                     *R_range, *z_range, *phi_range, size, use_physical)
    if dd:
        filename += '(dd)'
    filename += '.npy'
    
    if os.path.exists(filename):
        return np.load(filename)
    
    df = toomredf(n)
    samples = df.sample_cyl_selection(R_range, z_range, phi_range, selection, 
                                      size=size, use_physical=use_physical)
    samples = cyl_to_rect(*samples.T)
    
    if not os.path.exists('data'):
        os.mkdir('data')
    np.save(filename, samples)
    
    return samples

def sample_like_gaia(n, epsilon):
    gaia_data = search_local.search_phase_space(0, 0, 0, 0, 0, 0, epsilon, 0, 
                                                parallax_cut=False)
    R, z, phi = rect_to_cyl(*gaia_data.T).T[[0,3,5]]
    
    size = len(gaia_data)
    phi_range = [np.round(np.min(phi),2), np.round(np.max(phi),2)]
    R_range = [np.round(np.min(R),2), np.round(np.max(R),2)]
    
    return sample(n, R_range, [-24,24], phi_range, size=size, use_physical=True)

def sample_like_gaia_selection(n, epsilon):
    gaia_data = search_local.search_phase_space(0, 0, 0, 0, 0, 0, epsilon, 0, 
                                                parallax_cut=False)
    R, z, phi = rect_to_cyl(*gaia_data.T).T[[0,3,5]]
    
    size = len(gaia_data)
    phi_range = [np.round(np.min(phi),2), np.round(np.max(phi),2)]
    R_range = [np.round(np.min(R),2), np.round(np.max(R),2)]
    
    return sample_selection(n, R_range, [-1.5,1.5], phi_range, selection, 
                            size=size, use_physical=True)
    