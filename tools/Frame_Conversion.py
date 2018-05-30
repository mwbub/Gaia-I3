"""
NAME:
    frame conversion

PURPOSE:
    Use astropy to convert between different frames of coordinates
    
HISTORY:
    2018-05-30 - Written - Samuel Wong
"""
import numpy as np
from astropy.coordinates import SkyCoord, CartesianRepresentation, CartesianDifferential
import astropy.units as unit

def galactic_to_galactocentric(point):
    """
    NAME:
        galactic_to_galactocentric

    PURPOSE:
        Given 6 coordinates for the position and velocity of a star in
        galactic Cartesian coordinate, convert to galactocentric Cartesian

    INPUT:
        point = numpy array ([u, v, w, U, V, W]), where:
            u =  u coordinate in kpc
            v = v coordinate in kpc
            w = w coordinate in kpc
            U = velocity in U in km/s
            V = velocity in V in km/s
            W = velocity in W in km/s

    OUTPUT:
        numpy array ([x, y, z, vx, vy, vz]), where:
            x =  x coordinate in kpc
            y = y coordinate in kpc
            z = z coordinate in kpc
            vx = velocity in x in km/s
            vy = velocity in y in km/s
            vz = velocity in z in km/s

    HISTORY:
        2018-05-30 - Written - Samuel Wong
    """
    u, v, w, U, V, W = point
    coord = SkyCoord(frame = 'galactic', representation_type = CartesianRepresentation,
                 differential_type = CartesianDifferential,
                 u = u*unit.kpc, v = v*unit.kpc, w = w*unit.kpc,
                 U = U*unit.km/unit.s, V = V*unit.km/unit.s, W = W*unit.km/unit.s)
    coord = coord.transform_to('galactocentric')
    coord.representation_type = CartesianRepresentation
    x = coord.x.value
    y = coord.y.value
    z = coord.z.value
    vx = coord.v_x.value
    vy = coord.v_y.value
    vz = coord.v_z.value
    return np.array([x, y, z, vx, vy, vz])


def galactocentric_to_galactic(point):
    """
    NAME:
        galactocentric_to_galactic

    PURPOSE:
        Given 6 coordinates for the position and velocity of a star in
        galactocentric Cartesian coordinate, convert to galactic Cartesian

    INPUT:
        point = numpy array ([x, y, z, vx, vy, vz]), where:
            x =  x coordinate in kpc
            y = y coordinate in kpc
            z = z coordinate in kpc
            vx = velocity in x in km/s
            vy = velocity in y in km/s
            vz = velocity in z in km/s

    OUTPUT:            
        numpy array ([u, v, w, U, V, W]), where:
            u =  u coordinate in kpc
            v = v coordinate in kpc
            w = w coordinate in kpc
            U = velocity in U in km/s
            V = velocity in V in km/s
            W = velocity in W in km/s

    HISTORY:
        2018-05-30 - Written - Samuel Wong
    """
    x, y, z, vx, vy, vz = point
    coord = SkyCoord(frame = 'galactocentric', representation_type = CartesianRepresentation,
                 differential_type = CartesianDifferential,
                 x = x*unit.kpc, y = y*unit.kpc, z = z*unit.kpc,
                 v_x = vx*unit.km/unit.s, v_y = vy*unit.km/unit.s,
                 v_z = vz*unit.km/unit.s)
    coord = coord.transform_to('galactic')
    coord.representation_type = CartesianRepresentation
    u = coord.u.value
    v = coord.v.value
    w = coord.w.value
    U = coord.U.value
    V = coord.V.value
    W = coord.W.value
    return np.array([u, v, w, U, V, W])