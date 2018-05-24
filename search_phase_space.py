""" 
Filename: search6d.py
Author: Mathew Bub
Last Revision Date: 2018-05-22

This module contains the serach6d function, which queries the Gaia archive for
stars close to a given point in phase space, using a galacitc coordiante frame.
"""

from astroquery.gaia import Gaia
from galpy.util.bovy_coords import lb_to_radec
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

ra_ngp, dec_ngp = lb_to_radec(0, np.pi/2, epoch=None)
k = (u.kpc*u.mas/u.yr).to(u.km*u.rad/u.s)

def search_phase_space():
    import warnings
    warnings.filterwarnings("ignore")
    
    query = """
    SELECT *
    FROM (SELECT *,
          d * cosb * cosl as x,
          d * cosb * sinl as y,
          d * sinb as z,
          radial_velocity * cosb * cosl - {2} * d * pmb * sinb * cosl - {2} * d * pml_cosb * sinl as vx,
          radial_velocity * cosb * sinl - {2} * d * pmb * sinb * sinl + {2} * d * pml_cosb * cosl as vy,
          radial_velocity * sinb + {2} * d * pmb * cosb as vz
    FROM (SELECT *,
          pmra * cosphi + pmdec * sinphi as pml_cosb,
          pmdec * cosphi - pmra * sinphi as pmb
    FROM (SELECT *,
          SIN(RADIANS(ra) - {0}) * COS({1}) / cosb as sinphi,
          (SIN({1}) - sindec * sinb) / (cosdec * cosb) as cosphi
    FROM (SELECT TOP 5 *,
          SIN(RADIANS(dec)) as sindec,
          COS(RADIANS(dec)) as cosdec,
          SIN(RADIANS(b)) as sinb,
          COS(RADIANS(b)) as cosb,
          SIN(RADIANS(l)) as sinl,
          COS(RADIANS(l)) as cosl,
          1/parallax as d
    FROM gaiadr2.gaia_source
        WHERE 1=CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS',
        COORD1(EPOCH_PROP_POS(176.42881667,-64.84151667,215.7800,2662.0360,
                              -345.1830,0,2000,2015.5)),
        COORD2(EPOCH_PROP_POS(176.42881667,-64.84151667,215.7800,2662.0360,
                              -345.1830,0,2000,2015.5)), 1.0))
        AND radial_velocity IS NOT NULL) tab0) tab1) tab2) tab3
    """.format(ra_ngp, dec_ngp, k)
    job = Gaia.launch_job_async(query)
    return job.get_results()