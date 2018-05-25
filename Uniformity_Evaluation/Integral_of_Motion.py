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


def Energy(x, y, z, vx, vy, vz):
    R = np.sqrt(x**2 + y**2)
    phi = evaluatePotentials(MWPotential2014, R, z)
    kinetic = (vx**2 + vy**2 + vz**2)/2.
    energy = kinetic + phi
    return energy

def L_z(x,y,z,vx,vy,vz):
