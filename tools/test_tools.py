import numpy as np
from galpy.orbit import Orbit
from galpy.potential import MWPotential2014
from tools import *

def test_natural_unit():
    o = Orbit() # create sun's orbit
    o.turn_physical_off() # make galpy use natural coordinate
    ts = np.linspace(0,100,1000)
    o.integrate(ts,MWPotential2014)
    natural_coord = np.array([o.x(),o.y(),o.z(),o.vx(),o.vy(),o.vz()])
    
    o.turn_physical_on() # make galpy use physical coordinate
    physical_coord = np.array([o.x(),o.y(),o.z(),o.vx(),o.vy(),o.vz()])
    my_coord = to_natural_units(np.array([physical_coord]))[0]
    
    print('galpy natural coord = ', natural_coord)
    print('my natural coord = ', my_coord)
    
test_natural_unit()
