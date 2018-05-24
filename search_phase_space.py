""" 
Filename: search_phase_space.py
Author: Mathew Bub
Last Revision Date: 2018-05-24

This module contains the search_phase_space function, which queries the 
Gaia archive for stars close to a given point in phase space, using a galactic 
coordiante frame.
"""
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
from galpy.util.bovy_coords import lb_to_radec
from numpy import pi

ra_ngp, dec_ngp = lb_to_radec(0, pi/2, epoch=None)
k = (u.kpc*u.mas/u.yr).to(u.km*u.rad/u.s)

def search_phase_space(x, y, z, vx, vy, vz, epsilon, v_scale=1.0, cone_r=1.0):
    import warnings
    warnings.filterwarnings("ignore")
    
    x = u.Quantity(x, u.kpc)
    y = u.Quantity(y, u.kpc)
    z = u.Quantity(z, u.kpc)
    vx = u.Quantity(vx, u.km/u.s)
    vy = u.Quantity(vy, u.km/u.s)
    vz = u.Quantity(vz, u.km/u.s)
    
    gal_coord = SkyCoord(frame='galactic', representation_type='cartesian', 
                         u=x, v=y, w=z)
    icrs_coord = gal_coord.transform_to('icrs')
    cone_ra, cone_dec = icrs_coord.ra.value, icrs_coord.dec.value
    
    params = (k, dec_ngp, ra_ngp, cone_ra, cone_dec, cone_r, x.value, 
              y.value, z.value, vx.value, vy.value, vz.value, v_scale, epsilon)
    
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
    FROM (SELECT *,
          SIN(RADIANS(dec)) as sindec,
          COS(RADIANS(dec)) as cosdec,
          SIN(RADIANS(b)) as sinb,
          COS(RADIANS(b)) as cosb,
          SIN(RADIANS(l)) as sinl,
          COS(RADIANS(l)) as cosl,
          1/parallax as d,
          radial_velocity as vr
    FROM gaiadr2.gaia_source
        WHERE 1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {3}, {4}, {5})) 
        AND radial_velocity IS NOT NULL) tab0) tab1) tab2) tab3
    WHERE {13} > SQRT(POWER({6}-x,2) + POWER({7}-y,2) + POWER({8}-z,2) +
    (POWER({9}-vx,2) + POWER({10}-vy,2) + POWER({11}-vz,2))*POWER({12},2))
    """.format(*params)
    
    job = Gaia.launch_job_async(query)
    return job.get_results()