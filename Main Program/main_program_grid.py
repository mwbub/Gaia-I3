"""
NAME:
    main_program_grid

PURPOSE:
    choose a point in phase space and check whether the density is changing
    locally in the four dimensional plane where energy and angular momentum
    are conserved. If it is not, I_3 does not exist.
    
    Evaluate dot product on a 6 dimensional grid.
    
HISTORY:
    2018-06-04 - written - Michael Poon, Samuel Wong
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

def evaluate_uniformity_from_point(point_galactocentric, density):
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
    return directional_derivatives, del_E_a, del_Lz_a, W


def evaluate_uniformity_from_grid(density):
    # get a six dimensional grid to evaluate points at
    grid = create_meshgrid(xy_min, xy_max, xy_spacing, z_min, z_max, z_spacing,
                        vxy_min, vxy_max, vxy_spacing, vz_min, vz_max, vz_spacing)
    print(grid)

    # initialize an list of all directional derivatives for all stars
    list_directional_derivatives = []
    
    for i in range(len(grid)):
        # forbid user to evaluate energy at origin
        if (grid[i][0], grid[i][1], grid[i][2]) == (0.,0.,0.):
            print(grid[i])
            raise Exception("Cannot evaluate energy at origin")
            
        directional_derivative, del_E_a, del_Lz_a, W = evaluate_uniformity_from_point(grid[i], density)
        print("directional_derivative: ", directional_derivative)
        print((i+1), 'of 729')
        list_directional_derivatives.append(directional_derivative)
    
    # convert list to array and flatten the array
    list_directional_derivatives = np.array(list_directional_derivatives)
    list_directional_derivatives = np.concatenate(list_directional_derivatives)
    
    # print out important information from the result
    print('average of dot product = ', np.mean(list_directional_derivatives))
    print('maximum of dot product = ', np.max(list_directional_derivatives))
    print('minimum of dot product = ', np.min(list_directional_derivatives))
    print('standard deviation of dot product = ', np.std(list_directional_derivatives))


# define parameters for the search and KDE
epsilon = 0.2
v_scale = 0.1
width = 10
# define parameters for the grid
xy_min = -15
xy_max = 15
xy_spacing = 15
z_min = -0.15
z_max = 0.15
z_spacing = 0.15
vxy_min = -300
vxy_max = 300
vxy_spacing = 300
vz_min = -1
vz_max = 1
vz_spacing = 1

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
# display number of stars found
print('Found a sample of {} of stars.'.format(np.shape(samples)[0]))

# use the samples and a KDE learning method to generate a density function
density = generate_KDE(samples, 'epanechnikov', width)

evaluate_uniformity_from_grid(density)



