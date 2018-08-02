import sys
sys.path.append('..')

import numpy as np
from main.main_program_cluster import get_Energy_Lz_gradient
from check_uniformity_of_density.Integral_of_Motion import Energy, del_E
from check_uniformity_of_density.Uniformity_Evaluation import grad_multi
from galpy.potential import MWPotential2014, LogarithmicHaloPotential, \
    KeplerPotential, MiyamotoNagaiPotential, HernquistPotential, \
    evaluatePotentials

# The well-tested energy function from an old version of main
def old_Energy(coord):
    x, y, z, vx, vy, vz = coord.T
    R = np.sqrt(x**2 + y**2)
    potential = evaluatePotentials(MWPotential2014, R, z)
    kinetic = (vx**2 + vy**2 + vz**2)/2.
    energy = kinetic + potential
    return energy

# Set up a variety of potentials
pots = [MWPotential2014, LogarithmicHaloPotential(normalize=1.), 
        KeplerPotential(amp=1.), HernquistPotential(amp=1.,a=2.),
        MiyamotoNagaiPotential(a=0.5,b=0.0375,normalize=1.)]

# Get some random coordinates for testing
coords = np.random.uniform(low=0, high=2, size=(1000,6))

# Test that the new Energy correctly defaults to MWPotential2014
assert np.all(get_Energy_Lz_gradient(coords, 'analytic', None)[0] == 
              get_Energy_Lz_gradient(coords, 'analytic', MWPotential2014)[0])
assert np.all(get_Energy_Lz_gradient(coords, 'numeric', None)[0] ==
               get_Energy_Lz_gradient(coords, 'numeric', MWPotential2014)[0])

# Test that the old energy function has the same results as the new one
assert np.all(grad_multi(lambda x: Energy(x, None), coords) == 
              grad_multi(old_Energy, coords))

# Test different potentials
for pot in pots:
    # Test that the analytic energy gradient returns the same thing as del_E
    assert np.all(get_Energy_Lz_gradient(coords, 'analytic', pot)[0] == 
                  del_E(coords, pot))
    
    # Test that the numeric energy gradient returns the same thing as grad_multi
    assert np.all(get_Energy_Lz_gradient(coords, 'numeric', pot)[0] == 
                  grad_multi(lambda x: Energy(x, pot), coords))
    
    # Test that the analytic and numeric gradients are close
    assert np.all(np.isclose(get_Energy_Lz_gradient(coords, 'analytic', pot)[0],
                             get_Energy_Lz_gradient(coords, 'numeric', pot)[0],
                             atol = 1e-5, rtol=0))
    
# Test that different potentials yield different results for the gradient
for pot0 in pots:
    for pot1 in pots:
        if pot0 is not pot1:
            assert np.any(get_Energy_Lz_gradient(coords, 'analytic', pot0)[0] !=
                          get_Energy_Lz_gradient(coords, 'analytic', pot1)[0])
            
            assert np.any(get_Energy_Lz_gradient(coords, 'numeric', pot0)[0] !=
                          get_Energy_Lz_gradient(coords, 'numeric', pot1)[0])
                