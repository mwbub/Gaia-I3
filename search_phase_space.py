""" 
Filename: search6d.py
Author: Mathew Bub
Last Revision Date: 2018-05-22

This module contains the serach6d function, which queries the Gaia archive for
stars close to a given point in phase space, using a galacitc coordiante frame.
"""
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
from galpy.util.bovy_coords import lb_to_radec

ra_ngp, dec_ngp = lb_to_radec(0, np.pi/2, epoch=None)
k = (u.kpc*u.mas/u.yr).to(u.km*u.rad/u.s)

def search_phase_space(x, y, z, vx, vy, vz, epsilon, A):
    import warnings
    warnings.filterwarnings("ignore")
    
    query = """
    SELECT *
    FROM (SELECT *,
          d*cosb*cosl as x,
          d*cosb*sinl as y,
          d*sinb as z,
          vr*cosb*cosl - {0}*d*(pmb*sinb*cosl + pml_cosb*sinl) as vx,
          vr*cosb*sinl - {0}*d*(pmb*sinb*sinl - pml_cosb*cosl) as vy,
          vr*sinb + {0}*d*pmb*cosb as vz
    FROM (SELECT *,
          pmra*cosphi + pmdec*sinphi as pml_cosb,
          pmdec*cosphi - pmra*sinphi as pmb
    FROM (SELECT *,
          COS({1})*SIN(RADIANS(ra)-{2}) / cosb as sinphi,
          (SIN({1}) - sindec*sinb) / (cosdec*cosb) as cosphi
    FROM (SELECT TOP 5 *,
          SIN(RADIANS(dec)) as sindec,
          COS(RADIANS(dec)) as cosdec,
          SIN(RADIANS(b)) as sinb,
          COS(RADIANS(b)) as cosb,
          SIN(RADIANS(l)) as sinl,
          COS(RADIANS(l)) as cosl,
          1/parallax as d,
          radial_velocity as vr
    FROM gaiadr2.gaia_source
        WHERE radial_velocity IS NOT NULL) tab0) tab1) tab2) tab3
    WHERE SQRT(POWER(x-{3},2) + POWER(y-{4},2) + POWER(z-{5},2) +
    {9}*(POWER(vx-{6},2) + POWER(vy-{7},2) + POWER(vz-{8},2))) < {10}
    """.format(k, dec_ngp, ra_ngp, x, y, z, vx, vy, vz, A, epsilon)
    job = Gaia.launch_job_async(query)
    return job.get_results()

# (x-{3})**2 + (y-{4})**2 + (z-{5})**2 + {9}*((vx-{6})**2 + (vy-{7})**2 + (vz-{8})**2) 