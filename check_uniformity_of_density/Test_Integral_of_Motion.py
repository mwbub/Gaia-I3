from Integral_of_Motion import *
import numpy as np
from astropy.coordinates import SkyCoord, CartesianRepresentation, CartesianDifferential
import astropy.units as u
from galpy.orbit import Orbit
from galpy.potential import MWPotential2014

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
    
def test_energy_and_momentum(x,y,z,vx,vy,vz):
    e_initial = Energy(x,y,z,vx,vy,vz)
    R, phi, z, vR, vT, vz = cartesian_to_cylindrical(x, y, z, vx, vy, vz)
    o =  Orbit(vxvv = [R, vR, vT, z, vz, phi])
    ts = np.linspace(0,1000,100000)
    o.integrate(ts,MWPotential2014)
    t = 100
    e_final = Energy(o.x(t), o.y(t), o.z(t), o.vx(t), o.vy(t), o.vz(t))
    print('initial energy = {}, final energy = {}'.format(e_initial, e_final))
    
test_cartesian_to_cylindrical(x, y, z, vx, vy, vz)
print()
test_energy_and_momentum(x, y, z, vx, vy, vz)


"""
coord = SkyCoord(frame = 'galactic', representation_type = CartesianRepresentation,
                 differential_type = CartesianDifferential,
                 u = x*u.kpc, v = y*u.kpc, w = z*u.kpc, U = vx*u.km/u.s, 
                 V = vy*u.km/u.s, W = vz*u.km/u.s)
"""