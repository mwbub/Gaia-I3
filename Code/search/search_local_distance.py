""" 
Filename: search_local.py
Author: Mathew Bub
Last Revision Date: 2018-06-11

This module contains the search_phase_space function, which searches the 
Gaia archive for stars close to a given point in phase space, using a galactic 
coordinate frame. This version of the module uses a local downloaded copy of
the Gaia DR2 RV catalogue.
"""
import sys
sys.path.append('..')

import numpy as np
from astropy import units
from astropy.coordinates import SkyCoord
from tools import load

# check if the Gaia data has already been loaded on a previous run
try:
    _DISTANCE
    _DISTANCE_ERROR
    _GAIA_LOADED
    _PARALLAX_CUT
except NameError:
    _GAIA_LOADED = False
    _PARALLAX_CUT = None

def load_gaiarv(parallax_cut=True):
    """
    NAME:
        load_gaiarv
        
    PURPOSE:
        load the Gaia DR2 RV catalogue for use in search_local
        
    INPUT:
        parallax_cut - if True, will perform a cut for stars with parallax
        errors < 20% (optional; default = True)
        
    OUTPUT:
        None (defines global variables to store the Gaia data)
        
    WARNINGS:
        if using Spyder 3, this function works best with User Module Reloader
        disabled; otherwise, the Gaia data will have to be reloaded every
        time the search_local module is imported; the setting can be found in
        Tools -> Preferences -> Python interpreter -> User Module Reloader
    """
    global _DISTANCE
    global _DISTANCE_ERROR
    global _GAIA_LOADED
    global _PARALLAX_CUT
    
    # fields to load
    fields = ['parallax', 'parallax_error']
        
    # load the Gaia DR2 RV catalogue
    data = load.gaiarv(fields=fields, parallax_cut=parallax_cut)
        
    _DISTANCE = 1/data['parallax']
    _DISTANCE_ERROR = data['parallax_error']/data['parallax']**2
    
    # store the state of this load
    _GAIA_LOADED = True
    _PARALLAX_CUT = parallax_cut
    
def get_entire_catalogue(parallax_cut=True):
    """
    NAME:
        get_entire_catalogue
        
    PURPOSE:
        return the entire Gaia DR2 catalogue in galactocentric rectangular
        coordinates for use in generating a KDE
        
    INPUT:
        parallax_cut - if True, will perform a cut for stars with parallax
        errors < 20% (optional; default = True)
        
        return_frame - coordinate frame of the output; can be either
        'galactocentric' or 'galactic' (optional; default = 'galactocentric')
        
    OUTPUT:
        Nx6 array of rectangular phase space coordinates of the form 
        (x, y, z, vx, vy, vz) in [kpc, kpc, kpc, km/s, km/s, km/s]
    """
    # load the Gaia data if not already loaded or if the parallax_cut setting
    # of this search does not match the _PARALLAX_CUT of the loaded data
    if not _GAIA_LOADED or parallax_cut != _PARALLAX_CUT:
        load_gaiarv(parallax_cut=parallax_cut)
        
    return np.stack((_DISTANCE, _DISTANCE_ERROR), axis = 1)
