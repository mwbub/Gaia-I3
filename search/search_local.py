
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from gaia_tools import load

gaia_rv = load.gaiarv()

icrs_coord = SkyCoord(ra=gaia_rv['ra']*u.deg, 
                      dec=gaia_rv['dec']*u.deg,
                      distance=1/gaia_rv['parallax']*u.kpc,
                      pm_ra_cosdec=gaia_rv['pmra']*u.mas/u.yr,
                      pm_dec=gaia_rv['pmdec']*u.mas/u.yr,
                      radial_velocity=gaia_rv['radial_velocity']*u.km/u.s)

gal_coord = icrs_coord.transform_to('galactic')
gal_coord.representation_type = 'cartesian'