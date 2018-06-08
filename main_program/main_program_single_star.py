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
from check_uniformity_of_density.Integral_of_Motion import *
from check_uniformity_of_density.Linear_Algebra import *
from check_uniformity_of_density.Uniformity_Evaluation import *
from search import search_online
from kde_function.kde_function import *
from tools.tools import *


def evaluate_uniformity_from_point(point_galactocentric, density):
    """
    NAME:
        evaluate_uniformity_from_point

    PURPOSE:
        Given a density function and a point, find the gradient of energy
        and angular momentum and the density at the point, and find the normalize
        dot products between the gradient of density and four orthonormal basis
        vectors of the orthgonal complement of the gradient of energy and
        angular momentum.

    INPUT:
        point_galactocentric = the point in phase space with six coordinates 
                               in galactocentric Cartesian with numbers
                               representing units (kpc and km/s)
        density = a differentiable density function

    OUTPUT:
        directional_derivatives = a numpy array containing the directional
                                  derivative of f along each direction
                                  of the basis vectors generating the subspace

    HISTORY:
        2018-06-07 - Written - Samuel Wong
    """
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


def main(custom_density = None, search_method = "online"):
    # at this point, everything should have physical units
    # get coordinate of the star to be evaluated from user
    point_galactocentric, point_galactic = get_star_coord_from_user()
    if custom_density == None:
        # define parameters for the search and KDE
        epsilon = 0.2
        v_scale = 0.1
        # depending on the argument of main function, search stars online, locally
        # or use all of local catalogue
        # if we are searching, get stars within an epsilon ball of the point in 
        # phase space from Gaia, input the galactic coordinate into search function
        if search_method == "online":
            samples = search_online.search_phase_space(*point_galactic, epsilon, v_scale)
        elif search_method == "local":
            from search import search_local # only import local if needed, since it is slow
            samples = search_local.search_phase_space(*point_galactic, epsilon, v_scale)
        elif search_method == "all of local":
            from search import search_local # only import local if needed, since it is slow
            samples = search_local.get_entire_catalogue()
        print('Found a sample of {} of stars,'.format(np.shape(samples)[0]))
        # Turn all data to natrual units; working with natural unit, galactocentric,
        # cartesian from this point on
        samples = to_natural_units(samples)
        # use the samples and a KDE learning method to generate a density function
        density = generate_KDE(samples, 'epanechnikov', v_scale)
    else:
        density = custom_density # use the custom density function
    
    directional_derivatives = evaluate_uniformity_from_point(point_galactocentric, density)
    for i in range(len(directional_derivatives)):
        print('del_rho dot w_{} = {}'.format(i, directional_derivatives[i]))
    
if __name__ == "__main__":
    main(None, "local")
