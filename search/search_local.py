""" 
Filename: search_local.py
Author: Mathew Bub
Last Revision Date: 2018-06-04

This module contains the search_phase_space function, which searches the 
Gaia archive for stars close to a given point in phase space, using a galactic 
coordinate frame. This version of the module uses a local downloaded copy of
the Gaia DR2 RV catalogue.
"""
import numpy as np
from astropy import units
from astropy.coordinates import SkyCoord
from gaia_tools import load

if 'gaia_rv' not in globals():
    gaia_rv = load.gaiarv()

gaia_rv_icrs = SkyCoord(ra=gaia_rv['ra']*units.deg, 
                        dec=gaia_rv['dec']*units.deg,
                        distance=1/gaia_rv['parallax']*units.kpc,
                        pm_ra_cosdec=gaia_rv['pmra']*units.mas/units.yr,
                        pm_dec=gaia_rv['pmdec']*units.mas/units.yr,
                        radial_velocity=gaia_rv['radial_velocity']*units.km/units.s)

gaia_rv_gal = gaia_rv_icrs.transform_to('galactic')
gaia_rv_gal.representation_type = 'cartesian'

gaia_rv_galcen = gaia_rv_icrs.transform_to('galactocentric')
gaia_rv_galcen.representation_type = 'cartesian'

def search_phase_space(u0, v0, w0, U0, V0, W0, epsilon, v_scale=1.0):
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
        distances (optional; default = 1.0)
        
    OUTPUT:
        Nx6 array of galactocentric coordinates of the form 
        (x, y, z, vx, vy, vz) in [kpc, kpc, kpc, km/s, km/s, km/s],
        consisting of stars within a distance of epsilon from the point
        (u0, v0, w0, U0, V0, W0)
    """
    u0 = units.Quantity(u0, units.kpc).value
    v0 = units.Quantity(v0, units.kpc).value
    w0 = units.Quantity(w0, units.kpc).value
    U0 = units.Quantity(U0, units.km/units.s).value
    V0 = units.Quantity(V0, units.km/units.s).value
    W0 = units.Quantity(W0, units.km/units.s).value
    
    u = gaia_rv_gal.u.value
    v = gaia_rv_gal.v.value
    w = gaia_rv_gal.w.value
    U = gaia_rv_gal.U.value
    V = gaia_rv_gal.V.value
    W = gaia_rv_gal.W.value
    
    mask = ((u - u0)**2 + (v - v0)**2 + (w - w0)**2 + ((U - U0)**2 + 
            (V - V0)**2 + (W - W0)**2) * v_scale**2) < epsilon**2
             
    results = gaia_rv_galcen[mask]
    
    samples = np.stack([results.x.value, 
                        results.y.value, 
                        results.z.value, 
                        results.v_x.value, 
                        results.v_y.value, 
                        results.v_z.value], axis=1)
    
    if samples.size > 0:
        return samples
    raise Exception("no results found")
    
def get_entire_catalogue():
    """
    NAME:
        get_entire_catalogue
        
    PURPOSE:
        return the entire Gaia DR2 catalogue in galactocentric rectangular
        coordinates for use in generating a KDE
        
    INPUT:
        None
        
    OUTPUT:
        Nx6 array of galactocentric coordinates of the form 
        (x, y, z, vx, vy, vz) in [kpc, kpc, kpc, km/s, km/s, km/s]
    """
    samples = np.stack([gaia_rv_galcen.x.value,
                        gaia_rv_galcen.y.value,
                        gaia_rv_galcen.z.value,
                        gaia_rv_galcen.v_x.value,
                        gaia_rv_galcen.v_y.value,
                        gaia_rv_galcen.v_z.value], axis=1)
    return samples
             
    