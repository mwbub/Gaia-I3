"""
NAME:
    main_program_single_star

PURPOSE:
    choose a point in phase space and check whether the density is changing
    locally in the four dimensional plane where energy and angular momentum
    are conserved. If it is not, I_3 does not exist.
    
HISTORY:
    2018-05-28 - Written - Samuel Wong
    2018-05-31 - Changed to natural units - Samuel Wong
    2018-06-04 - Changed name from 'main_program' to 'main_program_single_star'
                - Samuel Wong
"""
import os, sys
# get the outer folder as the path
outer_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
check_uniformity_path =  os.path.abspath(os.path.join(outer_path, 'check_uniformity_of_density'))
sys.path.append(outer_path)
sys.path.append(check_uniformity_path)
# import relevant functions from different folders
from search_phase_space.search_phase_space import *
from check_uniformity_of_density.Integral_of_Motion import *
from check_uniformity_of_density.Linear_Algebra import *
from check_uniformity_of_density.Uniformity_Evaluation import *
from kde_function.kde_function import *
from tools.tools import *
    
# define parameters for the search and KDE
epsilon = 0.5
v_scale = 0.1
width = 10

# at this point, every thing should have physical units
# get coordinate of the star to be searched from user
point_galactocentric, point_galactic = get_star_coord_from_user()
# get stars within an epsilon ball of the point in phase space from Gaia
# input the galactic coordinate into search function
table = search_phase_space(*point_galactic, epsilon, v_scale)
 # convert from Gaia table to numpy array; output in galactocentric, with units
samples = table_to_samples(table)
# Turn all data to natrual units; working with natural unit, galactocentric,
# cartesian from this point on
samples = to_natural_units(samples)

# use the samples and a KDE learning method to generate a density function
density = generate_KDE(samples, 'epanechnikov', width)

# turn the galactocentric representation of the search star to be unit-less
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
directional_derivatives = evaluate_uniformity(density, a, W)
print('from a sample of {} of stars,'.format(np.shape(samples)[0]))
for i in range(len(directional_derivatives)):
    print('del_rho dot w_{} = {}'.format(i, directional_derivatives[i]))
