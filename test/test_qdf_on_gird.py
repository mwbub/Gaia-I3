"""
NAME:
    test_qdf

PURPOSE:
    Test the main function by replacing KDE with a quasiisothermal density
    function.
    
HISTORY:
    2018-06-01 - Written - Samuel Wong
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
#import qdf related things
from galpy.potential import MWPotential2014
from galpy.actionAngle import actionAngleAdiabatic
from galpy.df import quasiisothermaldf
aA= actionAngleAdiabatic(pot=MWPotential2014,c=True)

# set up qdf
qdf= quasiisothermaldf(1./3.,0.2,0.1,1.,1.,pot=MWPotential2014,aA=aA,cutcounter=True)

# define cartesian qdf
def cartesian_qdf(corrd):
    x, y, z, vx, vy, vz = corrd
    R, phi, z, vR, vT, vz = cartesian_to_cylindrical(x, y, z, vx, vy, vz)
    return qdf(R, vR, vT, z, vz)

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
    return directional_derivatives


def evaluate_uniformity_from_grid(density):
    # get a six dimensional grid to evaluate points at
    grid = create_meshgrid(xy_min, xy_max, xy_spacing, z_min, z_max, z_spacing,
                        vxy_min, vxy_max, vxy_spacing, vz_min, vz_max, vz_spacing)

    # initialize an list of all directional derivatives for all stars
    list_directional_derivatives = []
    
    for i in range(len(grid)):
        # forbid user to evaluate energy at origin
        if (grid[i][0], grid[i][1], grid[i][2]) == (0.,0.,0.):
            raise Exception("Cannot evaluate energy at origin")
        # evaluate uniformity at the point, which gives an array of 4 dot product
        directional_derivative = evaluate_uniformity_from_point(grid[i], density)
        list_directional_derivatives.append(directional_derivative)
    
    # convert list to array and flatten the array
    list_directional_derivatives = np.array(list_directional_derivatives)
    list_directional_derivatives = np.concatenate(list_directional_derivatives)
    
    # print out important information from the result
    print('average of dot product = ', np.mean(list_directional_derivatives))
    print('maximum of dot product = ', np.max(list_directional_derivatives))
    print('minimum of dot product = ', np.min(list_directional_derivatives))
    print('standard deviation of dot product = ', np.std(list_directional_derivatives))

# define parameters for the grid
xy_min = 0.5
xy_max = 1.5
xy_spacing = 0.5 # choose spacing such that xy is never 0
z_min = 0.5
z_max = 1.5
z_spacing = 0.5
vxy_min = 0.5
vxy_max = 1.5
vxy_spacing = 0.5
vz_min = 0.5
vz_max = 1.5
vz_spacing = 0.5

evaluate_uniformity_from_grid(cartesian_qdf)
    
    