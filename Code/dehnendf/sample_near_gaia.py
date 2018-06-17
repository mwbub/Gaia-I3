"""
Filename: sample_near_gaia.py
Author: Mathew Bub
Last Revision Date: 2018-06-17

This module contains tools for generating Dehnen DF sample data similar in size
and location to certain regions of the Gaia catalogue.
"""
import sys
sys.path.append('..')

import numpy as np
from sample import load_samples
from galpy.util.bovy_coords import rect_to_cyl
from search.search_local import search_phase_space

def get_sample_params(u0, v0, w0, epsilon, parallax_cut=True):
    """
    NAME:
        get_sample_params
        
    PURPOSE:
        return the number of stars, phi range, and radial range of stars from
        the Gaia catalogue that are within epsilon of the point (u0, v0, w0)
        
    INPUT:
        u0 - rectangular x coordinate in the galactic frame (can be Quantity,
        otherwise given in kpc)
        
        v0 - rectangular y coordinate in the galactic frame (can be Quantity,
        otherwise given in kpc)
        
        w0 - rectangular z coordinate in the galactic frame (can be Quantity,
        otherwise given in kpc)
        
        epsilon - radius in physical space in which to search for stars
        
        parallax_cut - if True, will perform a cut for stars with parallax
        errors < 20% (optional; default = True)
        
    OUTPUT:
        number of stars, phi range, and radial range
    """
    gaia_data = search_phase_space(u0, v0, w0, 0, 0, 0, epsilon, 0, 
                                   parallax_cut=parallax_cut)
    R, phi, z = rect_to_cyl(*gaia_data.T[:3])
    n = len(gaia_data)
    phi_range = [np.min(phi), np.max(phi)]
    r_range = [np.min(R), np.max(R)]
    
    return n, phi_range, r_range

def load_mock_data(u0, v0, w0, epsilon, parallax_cut=True):
    """
    NAME:
        load_mock_data
        
    PURPOSE:
        load mock Gaia data near the point (u0, v0, w0) sampled using Dehnen DF
        
    INPUT:
        u0 - rectangular x coordinate in the galactic frame (can be Quantity,
        otherwise given in kpc)
        
        v0 - rectangular y coordinate in the galactic frame (can be Quantity,
        otherwise given in kpc)
        
        w0 - rectangular z coordinate in the galactic frame (can be Quantity,
        otherwise given in kpc)
        
        epsilon - radius in physical space in which to search for stars
        
        parallax_cut - if True, will perform a cut for stars with parallax
        errors < 20% (optional; default = True)
        
    OUTPUT:
        nx6 array of rectangular galactocentric coordinates of the form 
        (x, y, z, vx, vy, vz) in [kpc, kpc, kpc, km/s, km/s, km/s],
        representing sampled stars
    """
    params = get_sample_params(u0, v0, w0, epsilon, parallax_cut=parallax_cut)
    return load_samples(*params)