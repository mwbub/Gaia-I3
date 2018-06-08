""" 
Filename: search_local.py
Author: Mathew Bub
Last Revision Date: 2018-06-08

This module contains the search_phase_space function, which searches the 
Gaia archive for stars close to a given point in phase space, using a galactic 
coordinate frame. This version of the module uses a local downloaded copy of
the Gaia DR2 RV catalogue.
"""
import os
import numpy as np
import astropy.io.fits as pyfits
from astropy import units
from astropy.coordinates import SkyCoord
from gaia_tools.load import path, download

try:
    gaiarv
    gaiarv_icrs
    gaiarv_gal
    gaiarv_galcen
except NameError:
    _GAIA_LOADED = False
else:
    _GAIA_LOADED = True

def load_gaiarv(parallax_cut=True, load_errors=False):
    global gaiarv
    global gaiarv_icrs
    global gaiarv_gal
    global gaiarv_galcen
    global _GAIA_LOADED
    
    # download the Gaia DR2 RV catalogue if not already downloaded
    file_paths= path.gaiarvPath()
    if not np.all([os.path.exists(file_path) for file_path in file_paths]):
        download.gaiarv()
        
    # fields to load
    fields = ['ra', 'dec', 'parallax', 'pmra', 'pmdec', 'radial_velocity']
        
    if parallax_cut:
        # load parallax error percentage
        fields.append('parallax_over_error')
        
    if load_errors:
        # load the errors for each phase space coordinate
        fields.extend(['ra_error', 'dec_error', 'parallax_error','pmra_error', 
                       'pmdec_error', 'radial_velocity_error'])
        
    # load the Gaia DR2 RV catalogue
    gaiarv = np.lib.recfunctions.stack_arrays(
        [np.array(pyfits.getdata(file_path, ext=1))[fields] for file_path 
         in file_paths], autoconvert=True)
    
    if parallax_cut:
        # cut out stars with parallax errors over 20%
        gaiarv = gaiarv[gaiarv['parallax_over_error'] > 5]
        
    # organize the catalogue into a SkyCoord object
    gaiarv_icrs = SkyCoord(ra=gaiarv['ra']*units.deg, 
                            dec=gaiarv['dec']*units.deg,
                            distance=1/gaiarv['parallax']*units.kpc,
                            pm_ra_cosdec=gaiarv['pmra']*units.mas/units.yr,
                            pm_dec=gaiarv['pmdec']*units.mas/units.yr,
                            radial_velocity=
                            gaiarv['radial_velocity']*units.km/units.s)
    
    # convert to galactic rectangular coordiantes
    gaiarv_gal = gaiarv_icrs.transform_to('galactic')
    gaiarv_gal.representation_type = 'cartesian'
    
    # convert to galactocentric rectangular coordiantes
    gaiarv_galcen = gaiarv_icrs.transform_to('galactocentric')
    gaiarv_galcen.representation_type = 'cartesian'
    
    _GAIA_LOADED = True

def search_phase_space(u0, v0, w0, U0, V0, W0, epsilon, v_scale):
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
        
    OUTPUT:
        Nx6 array of galactocentric coordinates of the form 
        (x, y, z, vx, vy, vz) in [kpc, kpc, kpc, km/s, km/s, km/s],
        consisting of stars within a distance of epsilon from the point
        (u0, v0, w0, U0, V0, W0)
    """
    if not _GAIA_LOADED:
        raise Exception("must call load_gaiarv before calling "
                        "search_phase_space")

    # convert coordinates into consistent units
    u0 = units.Quantity(u0, units.kpc).value
    v0 = units.Quantity(v0, units.kpc).value
    w0 = units.Quantity(w0, units.kpc).value
    U0 = units.Quantity(U0, units.km/units.s).value
    V0 = units.Quantity(V0, units.km/units.s).value
    W0 = units.Quantity(W0, units.km/units.s).value
    
    # grab the galactic coordinates of the Gaia RV catalogue
    u = gaiarv_gal.u.value
    v = gaiarv_gal.v.value
    w = gaiarv_gal.w.value
    U = gaiarv_gal.U.value
    V = gaiarv_gal.V.value
    W = gaiarv_gal.W.value
    
    # search for stars within a distance of epsilon from the point 
    # (u0, v0, w0, U0, V0, W0)
    mask = ((u - u0)**2 + (v - v0)**2 + (w - w0)**2 + ((U - U0)**2 + 
            (V - V0)**2 + (W - W0)**2) * v_scale**2) < epsilon**2
             
    # get the galactocentric coordinates of the stars that were found
    results = gaiarv_galcen[mask]
    
    # organize the coordinates into an Nx6 array
    samples = np.stack([results.x.value, 
                        results.y.value, 
                        results.z.value, 
                        results.v_x.value, 
                        results.v_y.value, 
                        results.v_z.value], axis=1)
    
    if len(samples) > 0:
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
    if not _GAIA_LOADED:
        raise Exception("must call load_gaiarv before calling "
                        "get_entire_cataloge")
    
    # organize the coordinates into an Nx6 array
    samples = np.stack([gaiarv_galcen.x.value,
                        gaiarv_galcen.y.value,
                        gaiarv_galcen.z.value,
                        gaiarv_galcen.v_x.value,
                        gaiarv_galcen.v_y.value,
                        gaiarv_galcen.v_z.value], axis=1)
    return samples
             