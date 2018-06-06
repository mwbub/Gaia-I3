""" 
Filename: search_phase_space.py
Author: Mathew Bub
Last Revision Date: 2018-06-04

This module contains the search_phase_space function, which queries the 
Gaia archive for stars close to a given point in phase space, using a galactic 
coordinate frame. Coordinate transformations are dervied from Bovy (2011).
[https://github.com/jobovy/stellarkinematics/blob/master/stellarkinematics.pdf]
"""
import numpy as np
from astropy import units
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
from galpy.util.bovy_coords import lb_to_radec

# ra and dec of the north galactic pole
ra_ngp, dec_ngp = lb_to_radec(0, np.pi/2, epoch=None)

# conversion factor from kpc*mas/yr to km/s
k = (units.kpc*units.mas/units.yr).to(units.km*units.rad/units.s)

def search_phase_space(u0, v0, w0, U0, V0, W0, epsilon, v_scale=1.0, 
                       cone_r=None):
    """
    NAME:
        search_phase_space
        
    PURPOSE:
        query the Gaia DR2 RV catalogue for stars near a point in phase space
        
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
        
        cone_r - cone search radius used to limit the initial size of the 
        query; if None, will use the minimal cone that completely encompasses
        the search sphere in physical space (optional; given in degrees;
        default = None)
        
    OUTPUT:
        astropy Table, containing stars from the Gaia DR2 RV catalogue that are
        within a distance of epsilon from the point (u0, v0, w0, U0, V0, W0)
    """
    import warnings
    warnings.filterwarnings("ignore")
    
    # convert coordinates into consistent units
    u0 = units.Quantity(u0, units.kpc)
    v0 = units.Quantity(v0, units.kpc)
    w0 = units.Quantity(w0, units.kpc)
    U0 = units.Quantity(U0, units.km/units.s)
    V0 = units.Quantity(V0, units.km/units.s)
    W0 = units.Quantity(W0, units.km/units.s)
    
    # distance check to limit the size of the initial query
    d = np.sqrt(u0.value**2 + v0.value**2 + w0.value**2)
    limiting_condition = "AND ABS({} - 1/parallax) < {}".format(d, epsilon)
    
    # add a cone search if the search sphere does not contain the Sun, or if 
    # cone_r is set manually
    if d > epsilon or cone_r is not None:
        
        # get ra and dec of the point (u0, v0, w0) for use in the cone search
        galactic_coord = SkyCoord(frame='galactic', u=u0, v=v0, w=w0,
                                  representation_type='cartesian')
        icrs_coord = galactic_coord.transform_to('icrs')
        cone_ra, cone_dec = icrs_coord.ra.value, icrs_coord.dec.value
        
        # calculate the minimal cone that will completely encompass the sphere
        if cone_r is None:
            h = d - epsilon**2 / d
            r = (epsilon / d)*np.sqrt(d**2 - epsilon**2)
            cone_r = np.degrees(np.arctan(r / h))
            
        # cone search to further limit the initial query
        limiting_condition += ("\n\tAND 1=CONTAINS(POINT('ICRS', ra, dec), "
                               "CIRCLE('ICRS', {}, {}, {}))"
                               ).format(cone_ra, cone_dec, cone_r)
    
    # query parameters
    params = (k, dec_ngp, ra_ngp, limiting_condition, u0.value, v0.value,
              w0.value, U0.value, V0.value, W0.value, v_scale, epsilon)
    
    # convert icrs coordinates to galactic rectangular coordinates, then query
    # for stars within a distance of epsilon from (u0, v0, w0, U0, V0, W0)
    query = """
    SELECT *
    FROM (SELECT *,
          d*cosb*cosl AS x,
          d*cosb*sinl AS y,
          d*sinb AS z,
          radial_velocity*cosb*cosl - ({0})*d*(pmb*sinb*cosl + pml*sinl) AS u,
          radial_velocity*cosb*sinl - ({0})*d*(pmb*sinb*sinl - pml*cosl) AS v,
          radial_velocity*sinb + ({0})*d*pmb*cosb AS w
    FROM (SELECT *,
          pmra*cosphi + pmdec*sinphi AS pml,
          pmdec*cosphi - pmra*sinphi AS pmb
    FROM (SELECT *,
          COS({1})*SIN(RADIANS(ra) - ({2})) / cosb AS sinphi,
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
        {3}) table0) table1) table2) table3
    WHERE POWER({11},2) > POWER({4}-x,2) + POWER({5}-y,2) + POWER({6}-z,2) +
    (POWER({7}-u,2) + POWER({8}-v,2) + POWER({9}-w,2))*POWER({10},2)
    """.format(*params)
    
    job = Gaia.launch_job_async(query)
    table = job.get_results()
    if table:
        return table
    raise Exception("query returned no results")

def table_to_samples(table):
    """
    NAME:
        table_to_samples
        
    PURPOSE:
        convert the table returned by search_phase_space into an Nx6 array
        of samples in galactocentric rectangular coordinates
        
    INPUT:
        table - astropy Table returned by search_phase_space
        
    OUTPUT:
        Nx6 array of galactocentric coordinates of the form 
        (x, y, z, vx, vy, vz) in [kpc, kpc, kpc, km/s, km/s, km/s]
    """
    icrs_coord = SkyCoord(ra=table['ra']*units.deg, 
                          dec=table['dec']*units.deg, 
                          distance=table['d']*units.kpc, 
                          pm_ra_cosdec=table['pmra']*units.mas/units.yr, 
                          pm_dec=table['pmdec']*units.mas/units.yr,
                          radial_velocity=
                          table['radial_velocity']*units.km/units.s)
    galcen_coord = icrs_coord.transform_to('galactocentric')
    galcen_coord.representation_type = 'cartesian'
    samples = np.stack([galcen_coord.x.value, 
                        galcen_coord.y.value, 
                        galcen_coord.z.value, 
                        galcen_coord.v_x.value, 
                        galcen_coord.v_y.value, 
                        galcen_coord.v_z.value], axis=1)
    return samples
