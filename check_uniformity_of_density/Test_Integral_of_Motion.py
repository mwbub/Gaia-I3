from galpy.potential import MWPotential2014
from Integral_of_Motion import cartesian_to_cylindrical
import numpy as np

# global variables
# for the position, I took an example from online, the answer is (2, pi/6, 4)
x = np.sqrt(3)
y = 1.
z = 4.
# for the velocity, I chose the test to see whether the total velocity is the same, which is clealry 1 here
vx = np.sqrt(1 / 3)
vy = np.sqrt(1 / 3)
vz = np.sqrt(1 / 3)


def test_cartesian_to_cylindrical(x, y, z, vx, vy, vz):
    R, phi, z, vR, vT, vz = cartesian_to_cylindrical(x, y, z, vx, vy, vz)
    print('cylindrical position: ({},{},{})'.format(R, phi, z))
    print('cylindrical velocity: ({},{},{})'.format(vR, vT, vz))
    v = np.sqrt(vR**2 + vT**2 + vz**2)
    print('net velocity = ', v)

test_cartesian_to_cylindrical(x, y, z, vx, vy, vz)

