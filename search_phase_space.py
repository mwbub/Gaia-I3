""" 
Filename: search6d.py
Author: Mathew Bub
Last Revision Date: 2018-05-22

This module contains the search_phase_space function, which queries the 
Gaia archive for stars close to a given point in phase space, using a galactic 
coordiante frame.
"""
import astropy.units as u
from astroquery.gaia import Gaia
from galpy.util.bovy_coords import lb_to_radec
from numpy import pi

ra_ngp, dec_ngp = lb_to_radec(0, pi/2, epoch=None)
k = (u.kpc*u.mas/u.yr).to(u.km*u.rad/u.s)

def search_phase_space(x, y, z, vx, vy, vz, v_scale, epsilon):
    import warnings
    warnings.filterwarnings("ignore")
    
    params = (k,
              dec_ngp,
              ra_ngp,
              u.Quantity(x, u.kpc).value,
              u.Quantity(y, u.kpc).value,
              u.Quantity(z, u.kpc).value,
              u.Quantity(vx, u.km/u.s).value,
              u.Quantity(vy, u.km/u.s).value,
              u.Quantity(vz, u.km/u.s).value,
              v_scale,
              epsilon)
    
    query = """
    SELECT *
    FROM (SELECT *,
          d*cosb*cosl as x,
          d*cosb*sinl as y,
          d*sinb as z,
          vr*cosb*cosl - ({0})*d*(pmb*sinb*cosl + pml_cosb*sinl) as vx,
          vr*cosb*sinl - ({0})*d*(pmb*sinb*sinl - pml_cosb*cosl) as vy,
          vr*sinb + ({0})*d*pmb*cosb as vz
    FROM (SELECT *,
          pmra*cosphi + pmdec*sinphi as pml_cosb,
          pmdec*cosphi - pmra*sinphi as pmb
    FROM (SELECT *,
          COS({1})*SIN(RADIANS(ra)-({2})) / cosb as sinphi,
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
    WHERE SQRT(POWER({3}-x,2) + POWER({4}-y,2) + POWER({5}-z,2) +
    (POWER({6}-vx,2) + POWER({7}-vy,2) + POWER({8}-vz,2))*POWER({9},2)) < {10}
    """.format(*params)
    
    job = Gaia.launch_job_async(query)
    return job.get_results()
