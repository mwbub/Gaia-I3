import numpy as np
from galpy.orbit import Orbit
from galpy.potential import MWPotential2014
from tools import *
import os, sys
outer_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(outer_path)
from check_uniformity_of_density.Integral_of_Motion import *

def test_natural_unit(t):
    o = Orbit() # create sun's orbit
    o.turn_physical_off() # make galpy use natural coordinate
    ts = np.linspace(0,100,1000)
    o.integrate(ts,MWPotential2014)
    natural_coord = np.array([o.x(t),o.y(t),o.z(t),o.vx(t),o.vy(t),o.vz(t)])
    natural_energy = o.E(t)
    natural_momentum = o.L(t)[0][2]
    
    o.turn_physical_on() # make galpy use physical coordinate
    physical_coord = np.array([o.x(t),o.y(t),o.z(t),o.vx(t),o.vy(t),o.vz(t)])
    my_coord = to_natural_units(np.array([physical_coord]))[0]
    my_energy = Energy(my_coord)
    my_momentum = L_z(my_coord)
    
    
    print('galpy natural coord = ', natural_coord)
    print('my natural coord = ', my_coord)
    print('galpy energy = ', natural_energy)
    print('my energy = ', my_energy)
    print('galpy natural momentum = ', natural_momentum)
    print('my momentum = ', my_momentum)
    
def test_frame_conversion(x, y, z, vx, vy, vz):
    point = np.array([x, y, z, vx, vy, vz])
    print(galactocentric_to_galactic(galactic_to_galactocentric(point)))
    print(galactic_to_galactocentric(galactocentric_to_galactic(point)))
    
    
    
test_natural_unit(38)
test_frame_conversion(1.,2.,3.5,6.7, 8.3,3.2)
