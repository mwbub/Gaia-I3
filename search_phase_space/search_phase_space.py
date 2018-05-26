""" 
Filename: search_phase_space.py
Author: Mathew Bub
Last Revision Date: 2018-05-24

This module contains the search_phase_space function, which queries the 
Gaia archive for stars close to a given point in phase space, using a galactic 
coordinate frame. Coordinate transformations are dervied from Bovy (2011).
[https://github.com/jobovy/stellarkinematics/blob/master/stellarkinematics.pdf]
"""
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
from galpy.util.bovy_coords import lb_to_radec
import numpy as np

# ra and dec of the north galactic pole
ra_ngp, dec_ngp = lb_to_radec(0, np.pi/2, epoch=None)

# conversion factor from kpc*mas/yr to km/s
k = (u.kpc*u.mas/u.yr).to(u.km*u.rad/u.s)

def search_phase_space(x, y, z, vx, vy, vz, epsilon, v_scale=1.0, cone_r=None):
    """
    NAME:
        search_phase_space
        
    PURPOSE:
        query the Gaia DR2 RV catalogue for stars near a point in phase space
        
    INPUT:
        x - rectangular x coordinate in the galactic frame (can be Quantity,
        otherwise given in kpc)
        
        y - rectangular y coordinate in the galactic frame (can be Quantity,
        otherwise given in kpc)
        
        z - rectangular z coordinate in the galactic frame (can be Quantity,
        otherwise given in kpc)
        
        vx - x velocity in the galactic frame (can be Quantity, otherwise given
        in km/s)
        
        vy - y velocity in the galactic frame (can be Quantity, otherwise given
        in km/s)
        
        vz - z velocity in the galactic frame (can be Quantity, otherwise given
        in km/s)
        
        epsilon - radius in phase space in which to search for stars
        
        v_scale - scale factor for velocities used when calculating phase space
        distances (optional; default = 1.0)
        
        cone_r - cone search radius used to limit the initial size of the query 
        (optional; given in degrees)
        
    OUTPUT:
        astropy Table, containing stars from the Gaia DR2 RV catalogue that are
        within a distance of epsilon from the point (x, y, z, vx, vy, vz)
        
    HISTORY:
        2018-05-24 - Written - Mathew Bub
    """
    import warnings
    warnings.filterwarnings("ignore")
    
    # convert coordinates into consistent units
    x = u.Quantity(x, u.kpc)
    y = u.Quantity(y, u.kpc)
    z = u.Quantity(z, u.kpc)
    vx = u.Quantity(vx, u.km/u.s)
    vy = u.Quantity(vy, u.km/u.s)
    vz = u.Quantity(vz, u.km/u.s)
    
    # get the ra and dec of the point (x, y, z), for use in a cone search
    # to limit the size of the intial query
    gal_coord = SkyCoord(frame='galactic', representation_type='cartesian', 
                         u=x, v=y, w=z)
    icrs_coord = gal_coord.transform_to('icrs')
    cone_ra, cone_dec = icrs_coord.ra.value, icrs_coord.dec.value
    
    d = np.sqrt(x.value**2 + y.value**2 + z.value**2)
    limiting_condition = ""
    if d > epsilon or cone_r is not None:
        if cone_r is None:
            h = d - epsilon**2 / d
            r = (epsilon / d)*np.sqrt(d**2 - epsilon**2)
            cone_r = np.degrees(np.arctan(r / h))
        limiting_condition += ("AND 1=CONTAINS(POINT('ICRS', ra, dec), "
                               "CIRCLE('ICRS', {}, {}, {}))\n\t"
                               ).format(cone_ra, cone_dec, cone_r)
    limiting_condition += "AND ABS({}-1/parallax) < {}".format(d, epsilon)
    
    # query parameters
    params = (k, dec_ngp, ra_ngp, limiting_condition, x.value, y.value,
              z.value, vx.value, vy.value, vz.value, v_scale, epsilon)
    
    # convert icrs coordinates to galactic rectangular coordinates, then query
    # for stars within a distance of epsilon from point (x, y, z, vx, vy, vz)
    query = """
    SELECT *
    FROM (SELECT *,
          d*cosb*cosl AS x,
          d*cosb*sinl AS y,
          d*sinb AS z,
          radial_velocity*cosb*cosl - ({0})*d*(pmb*sinb*cosl + pml*sinl) AS vx,
          radial_velocity*cosb*sinl - ({0})*d*(pmb*sinb*sinl - pml*cosl) AS vy,
          radial_velocity*sinb + ({0})*d*pmb*cosb AS vz
    FROM (SELECT *,
          pmra*cosphi + pmdec*sinphi AS pml,
          pmdec*cosphi - pmra*sinphi AS pmb
    FROM (SELECT *,
          COS({1})*SIN(RADIANS(ra)-({2})) / cosb AS sinphi,
          (SIN({1}) - sindec*sinb) / (cosdec*cosb) AS cosphi
    FROM (SELECT *,
          SIN(RADIANS(dec)) AS sindec,
          COS(RADIANS(dec)) AS cosdec,
          SIN(RADIANS(b)) AS sinb,
          COS(RADIANS(b)) AS cosb,
          SIN(RADIANS(l)) AS sinl,
          COS(RADIANS(l)) AS cosl,
          1/parallax AS d
    FROM gaiadr2.gaia_source
        WHERE radial_velocity IS NOT NULL
        {3}) tab0) tab1) tab2) tab3
    WHERE POWER({11},2) > POWER({4}-x,2) + POWER({5}-y,2) + POWER({6}-z,2) +
    (POWER({7}-vx,2) + POWER({8}-vy,2) + POWER({9}-vz,2))*POWER({10},2)
    """.format(*params)
    
    job = Gaia.launch_job_async(query)
    return job.get_results()