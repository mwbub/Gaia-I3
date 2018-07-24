"""
NAME:
    Integral_of_Motion

PURPOSE:
    This module contains energy and angular momentum functions, as well as 
    their gradients.
    It also contains cartesian to cylindrical conversion.

FUNCTIONS:
    Energy: Ths function takes the position and velocity of a star and returns its toal energy (per mass)
    L_z: This functiont takes the position and velocity of a star and returns its angular momentum (per mass)
         in the z-direction
    cartesian_to_cylindrical: Convert position and velocity

HISTORY:
    2018-05-25 - Written - Samuel Wong
    2018-07-10 - Added explicit gradient function - Samuel Wong
"""
import sys
sys.path.append("..")
from galpy.potential import MWPotential2014
from galpy.potential import evaluatePotentials
from galpy.potential import evaluatephiforces
from galpy.potential import evaluateRforces
from galpy.potential import evaluatezforces
import numpy as np
from tools.tools import rect_to_cyl, cyl_to_rect


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

def cylindrical_to_cartesian(R, phi, z, vR, vT, vz):
    """
    NAME:
        cylindrical_to_cartesian

    PURPOSE:
        Given 6 coordinates for the position and velocity of a star in
        cylindrical coordinate, convert to Cartesian

    INPUT:
        R = radius in cylindrical
        phi = angle from x axis
        z = z coordinate
        vR = radial velocity
        vT = tangential velocity
        vz = velocity in z

    OUTPUT:
        (x, y, z, vx, vy, vz) where:
        x =  x coordinate
        y = y coordinate
        z = z coordinate
        vx = velocity in x
        vy = velocity in y
        vz = velocity in z

    HISTORY:
        2018-05-25 - Written - Samuel Wong
    """
    x = R*np.cos(phi)
    y = R*np.sin(phi)
    vx = vR*np.cos(phi) - vT*np.sin(phi)
    vy = vT*np.cos(phi) + vR*np.sin(phi)
    return x, y, z, vx, vy, vz



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


def del_E(coord):
    """
    NAME:
        del_E

    PURPOSE:
        Given (m,6) array for a list of the position and velocity of stars in
        Cartesian coordinate, return the gradient vectors of energy in Cartesian
        form, in the corresponding row order.
        Assumes input and out put are in natrual unit.
        
    INPUT:
        coord = array([[x, y, z, vx, vy, vz], ...])
                where each row represents the coordinate of a star

    OUTPUT:
        del_E = gradient in Cartesian coordinate
                where each row represents the gradient of a star

    HISTORY:
        2018-07-10 - Written - Samuel Wong
        2018-07-24 - Changed to an array of points - Samuel Wong
    """
    x, y, z, vx, vy, vz = coord.T
    R, vR, vT, z, vz, phi = rect_to_cyl(x, y, z, vx, vy, vz)
    # get the force of the potential in cylindrical form
    F_phi = evaluatephiforces(MWPotential2014, R, z, phi)/R
    F_R = evaluateRforces(MWPotential2014, R, z, phi)
    F_z= evaluatezforces(MWPotential2014, R, z, phi)
    # return the gradient in Cartesian coordinate
    gradient = [F_phi*np.sin(phi) - F_R*np.cos(phi),
                -F_R*np.sin(phi)- F_phi*np.cos(phi), -F_z, vx, vy, vz]
    return np.array(gradient).T


def del_Lz(coord):
    """
    NAME:
        del_Lz

    PURPOSE:
        Given 6 coordinates for the position and velocity of a star in
        Cartesian coordinate, return the gradient vector of z - angular
        momentum in Cartesian form.
        Assumes input and out put are in natrual unit.
        
    INPUT:
        coord = array[(x, y, z, vx, vy, vz)]

    OUTPUT:
        del_Lz = gradient in Cartesian coordinate

    HISTORY:
        2018-07-10 - Written - Samuel Wong
    """
    x, y, z, vx, vy, vz = coord
    # return the gradient in Cartesian coordinate
    gradient = [vy, -vx, 0, -y, x, 0]
    return np.array(gradient)
    