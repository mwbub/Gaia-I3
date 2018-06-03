"""
NAME:
    test_fake_density_with_fake_energy

PURPOSE:
    Test the main function by replacing KDE with a fake density function. As
    well, test it with a known fake energy function. Since all the functions
    here are known analytic functions, the result should be very close to zero.
    
HISTORY:
    2018-06-03 - Written - Samuel Wong
"""
import os, sys
# get the outer folder as the path
outer_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
check_uniformity_path =  os.path.abspath(os.path.join(outer_path, 'check_uniformity_of_density'))
sys.path.append(outer_path)
sys.path.append(check_uniformity_path)
# import relevant functions from different folders
from check_uniformity_of_density.Integral_of_Motion import *
from check_uniformity_of_density.Linear_Algebra import *
from check_uniformity_of_density.Uniformity_Evaluation import *
from tools.tools import *


def third_function(coord):
    x, y, z, vx, vy, vz = coord
    #R, phi, z, vR, vT, vz = cartesian_to_cylindrical(x, y, z, vx, vy, vz)
    return x*y*z*vx*vy*vz
    

def fake_density(coord):
    e = fake_energy(coord)
    l = L_z(coord)
    return e**2 + l**2


def fake_density_2(coord):
    e = fake_energy(coord)
    l = L_z(coord)
    i3 = third_function(coord)
    return e**2 + l**2 + i3**2

def fake_energy(coord):
    x, y, z, vx, vy, vz = coord
    return vx**2 + vy**2 + vz**2 - x**2 - y**2 - z**2

# at this point, every thing should have physical units
# get coordinate of the star to be searched from user
point_galactocentric, point_galactic = get_star_coord_from_user()

# turn the galactocentric representation of the search star to be unit less
# rename the search star to a
a = to_natural_units(np.array([point_galactocentric]))[0]
# get the gradient of energy and momentum of the search star
del_E = grad(fake_energy, 6)
del_Lz = grad(L_z, 6)
del_E_a = del_E(a)
del_Lz_a = del_Lz(a)
# create matrix of the space spanned by direction of changing energy and momentum
V = np.array([del_E_a, del_Lz_a])
# get the 4 dimensional orthogonal complement of del E and del Lz
W = orthogonal_complement(V)
# evaluate if density is changing along the subspace 
# check to see if they are all 0; if so, it is not changing
directional_derivatives = evaluate_uniformity(fake_density, a, W)

for i in range(len(directional_derivatives)):
    print('del_rho dot w_{} = {}'.format(i, directional_derivatives[i]))

