"""
NAME:
    test_fake_density

PURPOSE:
    Test the main function by replacing KDE with a fake density function
    
HISTORY:
    2018-05-31 - Written - Samuel Wong
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
    

def fake_density(coord):
    e = Energy(coord)
    l = L_z(coord)
    return e**2 + l**2

# at this point, every thing should have physical units
# get coordinate of the star to be searched from user
point_galactocentric, point_galactic = get_star_coord_from_user()

# turn the galactocentric representation of the search star to be unit less
# rename the search star to a
a = to_natural_units(np.array([point_galactocentric]))[0]
# get the gradient of energy and momentum of the search star
del_E = grad(Energy, 6)
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
