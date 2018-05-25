"""
NAME:
    Integral_of_Motion

PURPOSE:
    This module contains energy and angular momentum functions.

FUNCTIONS:
    Energy: Ths function takes the position and velocity of a star and returns its toal energy (per mass)
    L_z: This functiont takes the position and velocity of a star and returns its angular momentum in the z-direction

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
        Given 6 coordinates for the position and velocity of a star in Cartesian coordinate, convert to cylindrical

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
        2018-05-24 - Written - Samuel Wong
    """
    R = np.sqrt(x**2 + y**2)
    phi = np.arctan(y/x)
    vR = (x * vx + y * vy) / np.sqrt(x ** 2 + y ** 2)
    vT = (x * vy - y * vx)/(x**2 + y**2)
    return R, phi, z, vR, vT, vz


def Energy(x, y, z, vx, vy, vz):
    R = np.sqrt(x**2 + y**2)
    phi = evaluatePotentials(MWPotential2014, R, z)
    kinetic = (vx**2 + vy**2 + vz**2)/2.
    energy = kinetic + phi
    return energy

#def L_z(x,y,z,vx,vy,vz):
    #R = np.sqrt(x ** 2 + y ** 2)


