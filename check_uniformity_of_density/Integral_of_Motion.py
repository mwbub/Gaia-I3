"""
NAME:
    Integral_of_Motion

PURPOSE:
    This module contains energy and angular momentum functions. It also contains cartesian to cylindrical conversion.

FUNCTIONS:
    Energy: Ths function takes the position and velocity of a star and returns its toal energy (per mass)
    L_z: This functiont takes the position and velocity of a star and returns its angular momentum (per mass)
         in the z-direction
    cartesian_to_cylindrical: Convert position and velocity

HISTORY:
    2018-05-25 - Written - Samuel Wong
"""
from galpy.potential import MWPotential2014
from galpy.potential import evaluatePotentials
import numpy as np


def cartesian_to_cylindrical(x, y, z, vx, vy, vz):
    """
    NAME:
        cartesian_to_cylindrical

    PURPOSE:
        Given 6 coordinates for the position and velocity of a star in
        Cartesian coordinate, convert to cylindrical

    INPUT:
        x =  x coordinate
        y = y coordinate
        z = z coordinate
        vx = velocity in x
        vy = velocity in y
        vz = velocity in z

    OUTPUT:
        (R, phi, z, vR, vT, vz) where:
        R = radius in cylindrical
        phi = angle from x axis
        z = z coordinate
        vR = radial velocity
        vT = tangential velocity
        vz = velocity in z

    HISTORY:
        2018-05-25 - Written - Samuel Wong
    """
    R = np.sqrt(x**2 + y**2)
    phi = np.arctan(y/x)
    vR = (x * vx + y * vy) / np.sqrt(x ** 2 + y ** 2)
    vT = R*(x * vy - y * vx)/(x**2 + y**2)
    return R, phi, z, vR, vT, vz


def Energy(coord):
    """
    NAME:
        Energy

    PURPOSE:
        Given 6 coordinates for the position and velocity of a star in Cartesian coordinate, return energy per mass
        Assumes input and out put are in natrual unit.
        
    INPUT:
        coord = (x, y, z, vx, vy, vz) = a numpy array of coordinate where:
        x =  x coordinate
        y = y coordinate
        z = z coordinate
        vx = velocity in x
        vy = velocity in y
        vz = velocity in z

    OUTPUT:
        energy = total energy per mass

    HISTORY:
        2018-05-25 - Written - Samuel Wong
    """
    x, y, z, vx, vy, vz = coord
    R, phi, z, vR, vT, vz = cartesian_to_cylindrical(x, y, z, vx, vy, vz)
    potential = evaluatePotentials(MWPotential2014, R, z)
    kinetic = (vx**2 + vy**2 + vz**2)/2.
    energy = kinetic + potential
    return energy


def L_z(coord):
    """
    NAME:
        L_z

    PURPOSE:
        Given 6 coordinates for the position and velocity of a star in Cartesian coordinate, return angular
         momentum around z-axis per mass.
         Assumes input and out put are in natrual unit.

    INPUT:
        coord = (x, y, z, vx, vy, vz) = a numpy array of coordinate where:
        x =  x coordinate
        y = y coordinate
        z = z coordinate
        vx = velocity in x
        vy = velocity in y
        vz = velocity in z

    OUTPUT:
        L_z = angular momentum around z-axis per mass

    HISTORY:
        2018-05-25 - Written - Samuel Wong
    """
    x, y, z, vx, vy, vz = coord
    # convert to cylindrical coordinate
    R, phi, z, vR, vT, vz = cartesian_to_cylindrical(x, y, z, vx, vy, vz)
    # evaluate the angular momentum
    result = R*vT
    return result
