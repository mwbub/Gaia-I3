"""
NAME:
    main_program_cluster

PURPOSE:
    Evaluate uniformity of dot product on a cluster of points in phase space
    given by kmeans.
    
HISTORY:
    2018-06-20 - Written - Samuel Wong
    2018-07-20 - Changed to analytic gradient - Samuel Wong
"""
import time as time_class
import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import os, sys
# get the outer folder as the path
sys.path.append('..')
sys.path.append('../check_uniformity_of_density')
# import relevant functions from different folders
from check_uniformity_of_density.Integral_of_Motion import *
from check_uniformity_of_density.Linear_Algebra import *
from check_uniformity_of_density.Uniformity_Evaluation import *
from search import search_online
from search import search_local
from kde.kde_function import *
from kmeans.kmeans import *
from tools.tools import *

# define parameters for the search and KDE as global variables
epsilon = 1.0
v_scale = 0.1

# create a subfolder to save results
if not os.path.exists('main_program_results'):
    os.mkdir('main_program_results')

def evaluate_uniformity_from_point(a, density, Energy_grad_fn, Lz_grad_fn):
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
        a = the point in phase space with six coordinates in galactocentric
            Cartesian with natural units
            
        density = a differentiable density function
        
        Energy_grad_fn = energy gradient function
        
        Lz_grad_fn = angular momentum gradient function

    OUTPUT:
        directional_derivatives = a numpy array containing the directional
                                  derivative of density along each direction
                                  of the basis vectors generating the subspace

    HISTORY:
        2018-06-20 - Written - Samuel Wong
        2018-07-15 - Added gradient functions as input - Samuel Wong
    """
    # get the gradient of energy and momentum of the search star
    del_E_a = Energy_grad_fn(a)
    del_Lz_a = Lz_grad_fn(a)
    # create matrix of the space spanned by direction of changing energy and momentum
    V = np.array([del_E_a, del_Lz_a])
    # get the 4 dimensional orthogonal complement of del E and del Lz
    W = orthogonal_complement(V)
    # evaluate if density is changing along the subspace 
    # check to see if they are all 0; if so, it is not changing
    directional_derivatives = evaluate_uniformity(density, a, W)
    return directional_derivatives


def search_for_samples(search_method):
    """
    NAME:
        search_for_samples

    PURPOSE:
        Depending on the argument, search_method, search stars online, locally
        or use all of local catalogue. If we are searching, get stars within an 
        epsilon ball of the point in phase space from Gaia. We always input the
        galactic coordinate into search function. In addition, generate
        a file name to save result

    INPUT:
        search_method = a string that is either "online", "local", or
                        "all of local"

    OUTPUT:
        samples = a numpy arrays containing 6 dimensional coordinates in
                  galactocentric Cartesian form with physical units
                  
        file_name = a string that records the epsilon, v_scale, and search star
                    or the fact that all of Gaia catalogue was used

    HISTORY:
        2018-06-25 - Written - Samuel Wong
    """
    if search_method == "online":
        # get coordinate of the star to be searched from user
        point_galactocentric, point_galactic = get_star_coord_from_user()
        samples = search_online.search_phase_space(*point_galactic, epsilon, v_scale)
    elif search_method == "local":
        # get coordinate of the star to be searched from user
        point_galactocentric, point_galactic = get_star_coord_from_user()
        samples = search_local.search_phase_space(*point_galactic, epsilon, v_scale)
    elif search_method == "all of local":
        samples = search_local.get_entire_catalogue()
    print('Found a sample of {} of stars,'.format(np.shape(samples)[0]))
    
    # create file name if a search was performed
    # if user is using all of catalogue, record this fact and epsilon and
    # v_scale in file name
    if search_method == "all of local":
        file_name = 'full sample'
    # if actual search was done, record search star as well
    else:
        file_name = 'epsilon = {}, v_scale = {}, star galactocentric = {}'.format(
                epsilon, v_scale, np.array_str(point_galactocentric))
    # remove any line with \n in the title
    file_name = file_name.replace('\n','')
    
    return samples, file_name


def get_samples_density_filename(custom_density, search_method, custom_samples):
    """
    NAME:
        get_samples_density_filename

    PURPOSE:
        Given the input of the main functions, which are custom density,
        search method, and custom samples, check whether custom density or 
        custom samples are None. If they are, search for stars in Gaia and use
        KDE to generate density. If one of them is not None, use the custom
        density or custom samples (to generate density through KDE). Also,
        generate appropriate file name and create a subfolder to save file.

    INPUT:
        custom_density = a function that takes 6 values in an array and outputs
                         the density at that point; can also be None
        search_method = one of "online", "local", "all of local"
        custom_samples = an N by 6 arrays in physical units representing 6
                         dimensional star coordinate; can also be None

    OUTPUT:
        samples = either custom or searched
        density = density function, either custom or generated by KDE
        file_name = a string representing the initial values for search, or
                    user provided if no search was done

    HISTORY:
        2018-06-25 - Written - Samuel Wong
    """   
    # use custom samples or search for samples in Gaia
    if custom_samples is not None:
        file_name = input('Name of file to be saved: ')
        samples = custom_samples
    else:
        samples, file_name = search_for_samples(search_method) 
        
    # at this point, everything should have physical units
    # turn all data to natrual units; working with natural unit, galactocentric,
    # cartesian from this point on
    samples = to_natural_units(samples)
        
    # use a custom density or generate a density using a KDE
    if custom_density is not None:
        density = custom_density
        name_of_density = input('Name of custom density function: ')
        file_name = name_of_density + ' ' + file_name
    else:
        density = generate_KDE(samples, 'epanechnikov')
        
    # create a sub-subfolder to save results
    if not os.path.exists('main_program_results/'+file_name):
        os.mkdir('main_program_results/'+file_name)
        
    return samples, density, file_name


def get_cluster(samples):
    """
    NAME:
        get_cluster

    PURPOSE:
        Given a sample, use kmeans minibatch to find some cluster centers that
        are a good representative of the samples.

    INPUT:
        samples = a numpy arrays containing 6 dimensional coordinates in
                  galactocentric Cartesian form with natural units

    OUTPUT:
        cluster = a numpy arrays containing 6 dimensional coordinates in
                  galactocentric Cartesian form with natural units; a smal but 
                  representative cluster centers of samples

    HISTORY:
        2018-06-25 - Written - Samuel Wong
    """
    # let batch size be 1% of the number of samples
    batch_size = int(0.01 * np.shape(samples)[0])
    # let the number of cluster centers to be 0.1% of number of samples
    cluster_number = int(0.001 * np.shape(samples)[0])
    # use kmenas to generate a cluster of points
    cluster = kmeans(samples, cluster_number, batch_size)
    return cluster


def main(custom_density = None, search_method = "local", custom_samples = None,
         gradient_method = "analytic", custom_centres=None):
    """
    NAME:
        main

    PURPOSE:
        Call on all modules to evaluate uniformity of density on a cluster of 
        points provided by kmeans. Allows the user to specify search method to
        generate sample stars around a point in phase space. Also, allows user
        to give custom density function, but since no points are given for a
        custom density, this program only evaluates uniformity at a point when
        custom density is given. Also, allows user to give custom sample stars.
        Output figures of results, including kmeans and dot product scatter
        plot.

    INPUT:
        custom_density = a customized density functiont that takes an array
                         of 6 numbers representing the coordinate and return
                         the density; if this input is None, then the code
                         will use a search method to get data from Gaia catalogue
                         and use KDE to genereate a density function
        search_method = search the gaia catalogue online ("online"),
                        locally on a downloaded file ('local'), or use the
                        the entire downloaded gaia rv file ('all of local')
        custom_samples = an N by 6 array that represents the custom samples, 
                        with each component representing (x,y,z,vx,vy,vz),
                        respectively. They are in physical units.
        gradient_method = "analytic" or "numeric", referring to how gradient
                          of energy and L_z are generated
        custom_centres = a custom array of cluster centres at which to evaluate
                         uniformity; if None, will use kmeans clustering to get
                         the cluster centres

    HISTORY:
        2018-06-20 - Written - Samuel Wong
        2018-06-21 - Added option of custom samples - Samuel Wong and Michael
                                                      Poon
        2018-06-22 - Added Figure
        2018-07-15 - Added choice of gradient method
    """        
    samples, density, file_name = get_samples_density_filename(
            custom_density, search_method, custom_samples)    
    
    # use kmeans to get cluster centres or use custom centres
    if custom_centres is not None:
        cluster = custom_centres
    else:
        cluster = get_cluster(samples)
    
    # set the gradient function of energy and L_z based on specified method
    if gradient_method == "analytic":
        Energy_grad_fn = del_E
        Lz_grad_fn = del_Lz
    elif gradient_method == "numeric":
        Energy_grad_fn = grad(Energy, 6)	
        Lz_grad_fn = grad(L_z, 6)
    
    # initialize an array of directional derivative for each point
    result = np.empty((np.shape(cluster)[0], 4))
    # evaluate uniformity for each point in cluster
    start = time_class.time()
    for (i, point) in enumerate(cluster):
        result[i] = evaluate_uniformity_from_point(point, density, 
              Energy_grad_fn, Lz_grad_fn)
        print('At point {}, dot products are {}'.format(point, result[i]))
        print()
    inter_time = time_class.time() - start
    print('time per star =', inter_time/np.shape(cluster)[0])
    # output summary information
    # report the average and standard deviation of the maximum 
    # dot product in absolute value, ignoring nan values
    max_dot_product = np.nanmax(np.absolute(result), axis = 1)
    mean_of_max = np.nanmean(max_dot_product)
    std_of_max = np.nanstd(max_dot_product, ddof = 1)
    print('The average of the maximum absolute value of dot product is ', mean_of_max)
    print('The standard deviation of the maximum absolute value of dot product is ', std_of_max)
    # save result
    np.savez('main_program_results/' + file_name +'/'+ 'data', 
             cluster = cluster, result = result)
    
    # create and save graph of kmeans projection in 2 dimension
    kmeans_plot(samples, cluster, file_name)
    # create and save graph of dot product
    dot_product_plot(max_dot_product, cluster, file_name)
    #create and save graph of dot product in color scatter plot in all 
    # 2 dimensional projection angles.
    color_plot(max_dot_product, cluster, file_name)
        
    
if __name__ == "__main__":
    main(custom_density = None, search_method = "local", custom_samples = None,
         gradient_method = "analytic")
