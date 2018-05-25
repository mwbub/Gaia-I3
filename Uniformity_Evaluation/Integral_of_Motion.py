

from galpy.potential import MWPotential2014
from galpy.potential import evaluatePotentials
import numpy as np


def Energy(x, y, z, vx, vy, vz):
    R = np.sqrt(x**2 + y**2)
    phi = evaluatePotentials(MWPotential2014, R, z)
    kinetic = (vx**2 + vy**2 + vz**2)/2.
    energy = kinetic + phi
    return energy
