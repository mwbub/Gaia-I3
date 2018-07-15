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
    _GAIARV_GAL
    _GAIARV_GALCEN
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
    global _GAIARV_GAL
    global _GAIARV_GALCEN
    global _GAIA_LOADED
    global _PARALLAX_CUT
    
    # fields to load
    fields = ['ra', 'dec', 'parallax', 'pmra', 'pmdec', 'radial_velocity']
        
    # load the Gaia DR2 RV catalogue
    data = load.gaiarv(fields=fields, parallax_cut=parallax_cut)
        
    # organize the catalogue into a SkyCoord object
    gaiarv_icrs = SkyCoord(ra=data['ra']*units.deg,
                           dec=data['dec']*units.deg,
                           distance=1/data['parallax']*units.kpc,
                           pm_ra_cosdec=data['pmra']*units.mas/units.yr,
                           pm_dec=data['pmdec']*units.mas/units.yr,
                           radial_velocity=
                           data['radial_velocity']*units.km/units.s)
    
    # convert to galactic rectangular coordiantes
    _GAIARV_GAL = gaiarv_icrs.transform_to('galactic')
    _GAIARV_GAL.representation_type = 'cartesian'
    
    # convert to galactocentric rectangular coordiantes
    _GAIARV_GALCEN = gaiarv_icrs.transform_to('galactocentric')
    _GAIARV_GALCEN.representation_type = 'cartesian'
    
    # store the state of this load
    _GAIA_LOADED = True
    _PARALLAX_CUT = parallax_cut

def search_phase_space(u0, v0, w0, U0, V0, W0, epsilon, v_scale,
                       parallax_cut=True, return_frame='galactocentric'):
    """
    NAME:
        search_phase_space
    
    PURPOSE:
        search the Gaia DR2 RV catalogue for stars near a point in phase space
        
    INPUT:
        u0 - rectangular x coordinate in the galactic frame (can be Quantity,
        otherwise given in kpc)
        
        v0 - rectangular y coordinate in the galactic frame (can be Quantity,
        otherwise given in kpc)
        
        w0 - rectangular z coordinate in the galactic frame (can be Quantity,
        otherwise given in kpc)
        
        U0 - x velocity in the galactic frame (can be Quantity, otherwise given
        in km/s)
        
        V0 - y velocity in the galactic frame (can be Quantity, otherwise given
        in km/s)
        
        W0 - z velocity in the galactic frame (can be Quantity, otherwise given
        in km/s)
        
        epsilon - radius in phase space in which to search for stars
        
        v_scale - scale factor for velocities used when calculating phase space
        distances
        
        parallax_cut - if True, will perform a cut for stars with parallax
        errors < 20% (optional; default = True)
        
        return_frame - coordinate frame of the output; can be either
        'galactocentric' or 'galactic' (optional; default = 'galactocentric')
        
    OUTPUT:
        Nx6 array of rectangular phase space coordinates of the form 
        (x, y, z, vx, vy, vz) in [kpc, kpc, kpc, km/s, km/s, km/s],
        consisting of stars within a distance of epsilon from the point
        (u0, v0, w0, U0, V0, W0)
    """
    import warnings
    warnings.filterwarnings('ignore')
    
    # load the Gaia data if not already loaded or if the parallax_cut setting
    # of this search does not match the _PARALLAX_CUT of the loaded data
    if not _GAIA_LOADED or parallax_cut != _PARALLAX_CUT:
        load_gaiarv(parallax_cut=parallax_cut)

    # convert coordinates into consistent units
    u0 = units.Quantity(u0, units.kpc).value
    v0 = units.Quantity(v0, units.kpc).value
    w0 = units.Quantity(w0, units.kpc).value
    U0 = units.Quantity(U0, units.km/units.s).value
    V0 = units.Quantity(V0, units.km/units.s).value
    W0 = units.Quantity(W0, units.km/units.s).value
    
    # grab the galactic coordinates of the Gaia RV catalogue
    u = _GAIARV_GAL.u.value
    v = _GAIARV_GAL.v.value
    w = _GAIARV_GAL.w.value
    U = _GAIARV_GAL.U.value
    V = _GAIARV_GAL.V.value
    W = _GAIARV_GAL.W.value
    
    # search for stars within a distance of epsilon from the point 
    # (u0, v0, w0, U0, V0, W0)
    mask = ((u - u0)**2 + (v - v0)**2 + (w - w0)**2 + ((U - U0)**2 + 
            (V - V0)**2 + (W - W0)**2) * v_scale**2) < epsilon**2
             
    if return_frame == 'galactocentric':
        # get the galactocentric coordinates of the stars that were found
        results = _GAIARV_GALCEN[mask]
        
        # organize the coordinates into an Nx6 array
        samples = np.stack([results.x.value, 
                            results.y.value, 
                            results.z.value, 
                            results.v_x.value, 
                            results.v_y.value, 
                            results.v_z.value], axis=1)
    elif return_frame == 'galactic':
        # get the galactic coordinates of the stars that were found
        results = _GAIARV_GAL[mask]
        
        # organize the coordinates into an Nx6 array
        samples = np.stack([results.u.value, 
                            results.v.value, 
                            results.w.value, 
                            results.U.value, 
                            results.V.value, 
                            results.W.value], axis=1)
    else:
        raise ValueError("return_frame must be 'galactocentric' or 'galactic'")
        
    if len(samples) > 0:
        return samples
    raise Exception('no results found')
    
def get_entire_catalogue(parallax_cut=True, return_frame='galactocentric'):
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
    
    if return_frame == 'galactocentric':
        # organize the coordinates into an Nx6 array
        samples = np.stack([_GAIARV_GALCEN.x.value,
                            _GAIARV_GALCEN.y.value,
                            _GAIARV_GALCEN.z.value,
                            _GAIARV_GALCEN.v_x.value,
                            _GAIARV_GALCEN.v_y.value,
                            _GAIARV_GALCEN.v_z.value], axis=1)
    elif return_frame == 'galactic':
        # organize the coordinates into an Nx6 array
        samples = np.stack([_GAIARV_GAL.u.value,
                            _GAIARV_GAL.v.value,
                            _GAIARV_GAL.w.value,
                            _GAIARV_GAL.U.value,
                            _GAIARV_GAL.V.value,
                            _GAIARV_GAL.W.value], axis=1)
    else:
        raise ValueError("return_frame must be 'galactocentric' or 'galactic'")
        
    return samples
