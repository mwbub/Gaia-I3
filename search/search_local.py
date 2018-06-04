
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from gaia_tools import load

if 'gaia_rv' not in globals():
    gaia_rv = load.gaiarv()

gaia_rv_icrs = SkyCoord(ra=gaia_rv['ra']*u.deg, 
                      dec=gaia_rv['dec']*u.deg,
                      distance=1/gaia_rv['parallax']*u.kpc,
                      pm_ra_cosdec=gaia_rv['pmra']*u.mas/u.yr,
                      pm_dec=gaia_rv['pmdec']*u.mas/u.yr,
                      radial_velocity=gaia_rv['radial_velocity']*u.km/u.s)

gaia_rv_gal = gaia_rv_icrs.transform_to('galactic')
gaia_rv_gal.representation_type = 'cartesian'

gaia_rv_galcen = gaia_rv_icrs.transform_to('galactocentric')
gaia_rv_galcen.representation_type = 'cartesian'

def search_phase_space(x0, y0, z0, U0, V0, W0, epsilon, v_scale=1.0):
    x0 = u.Quantity(x0, u.kpc).value
    y0 = u.Quantity(y0, u.kpc).value
    z0 = u.Quantity(z0, u.kpc).value
    U0 = u.Quantity(U0, u.km/u.s).value
    V0 = u.Quantity(V0, u.km/u.s).value
    W0 = u.Quantity(W0, u.km/u.s).value
    
    x = gaia_rv_gal.u.value
    y = gaia_rv_gal.v.value
    z = gaia_rv_gal.w.value
    U = gaia_rv_gal.U.value
    V = gaia_rv_gal.V.value
    W = gaia_rv_gal.W.value
    
    mask = ((x - x0)**2 + (y - y0)**2 + (z - z0)**2 + ((U - U0)**2 + 
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
    
    
    
             
    