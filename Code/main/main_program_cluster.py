"""
NAME:
    main_program_cluster

PURPOSE:
    Evaluate uniformity of dot product on a cluster of points in phase space
    given by kmeans.
    
HISTORY:
    2018-06-20 - Written - Samuel Wong
"""
import time as time_class
import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
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
from search import search_local
from kde.kde_function import *
from kmeans.kmeans import *
from tools.tools import *

# declare gradient functions for energy and momentum as global variables
del_E = grad(Energy, 6)
del_Lz = grad(L_z, 6)
# define parameters for the search and KDE as global variables
epsilon = 0.5
v_scale = 0.1

def evaluate_uniformity_from_point(a, density):
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

    OUTPUT:
        directional_derivatives = a numpy array containing the directional
                                  derivative of density along each direction
                                  of the basis vectors generating the subspace

    HISTORY:
        2018-06-20 - Written - Samuel Wong
    """
    # get the gradient of energy and momentum of the search star
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


def main(custom_density = None, search_method = "local", custom_samples = None):
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

    HISTORY:
        2018-06-20 - Written - Samuel Wong
        2018-06-21 - Added option of custom samples - Samuel Wong and Michael
                                                      Poon
        2018-06-22 - Added Figure
    """
    #  first, check whether custom density is given. If no, then see what
    # samples to generate density
    if custom_density == None:
        # second, if no custom samples are given, then search Gaia and generate
        # KDE
        if np.any(custom_samples == None):
            # get coordinate of the star to be searched from user
            # at this point, everything should have physical units
            point_galactocentric, point_galactic = get_star_coord_from_user()
            # depending on the argument of main function, search stars online, locally
            # or use all of local catalogue
            # if we are searching, get stars within an epsilon ball of the point in 
            # phase space from Gaia, input the galactic coordinate into search function
            if search_method == "online":
                samples = search_online.search_phase_space(*point_galactic, epsilon, v_scale)
            elif search_method == "local":
                samples = search_local.search_phase_space(*point_galactic, epsilon, v_scale)
            elif search_method == "all of local":
                samples = search_local.get_entire_catalogue()
            print('Found a sample of {} of stars,'.format(np.shape(samples)[0]))
        # But, if there is custom samples, then set that to be the samples
        else:
            samples = custom_samples
            # since custom samples are used, ask user for the name of the file
            # before running
            file_name = input('Name of file to be saved: ')
            # set search method to None so that future file name evaluation 
            # will not be triggered
            search_method = None
        # Turn all data to natrual units; working with natural unit, galactocentric,
        # cartesian from this point on
        samples = to_natural_units(samples)
        # use the samples and a KDE learning method to generate a density function
        density = generate_KDE(samples, 'epanechnikov', v_scale)
    # if density is given, use it directly, skipping all steps with search and
    # KDE
    else:
        density = custom_density # use the custom density function
        # get coordinate of the star to be evaluated from user
        # at this point, everything should have physical units
        point_galactocentric, point_galactic = get_star_coord_from_user()
    
    # if custom density is given, only evaluate uniformity at given point
    if custom_density != None:
        # convert the point to natural unit first
        point_gc_natural = to_natural_units(np.array([point_galactocentric]))[0]
        directional_derivatives = evaluate_uniformity_from_point(
                point_gc_natural, density)
        for i in range(len(directional_derivatives)):
            print('del_rho dot w_{} = {}'.format(i, directional_derivatives[i]))
    else:
        # let batch size be 10% of the number of samples
        batch_size = int(0.1 * np.shape(samples)[0])
        # let the number of cluster centers to be 1% of number of samples
        cluster_number = int(0.01 * np.shape(samples)[0])
        # use kmenas to generate a cluster of points
        cluster = kmeans(samples, cluster_number, batch_size)
        # initialize an array of directional derivative for each point
        result = np.empty((np.shape(cluster)[0], 4))
        # evaluate uniformity for each point in cluster
        start = time_class.time()
        for (i, point) in enumerate(cluster):
            result[i] = evaluate_uniformity_from_point(point, density)
            print('At point {}, dot products are {}'.format(point, result[i]))
            print()
        inter_time = time_class.time() - start
        print('time per star =', inter_time/cluster_number)
        # output summary information
        # report the average and standard deviation of the maximum 
        # dot product in absolute value, ignoring nan values
        max_dot_product = np.nanmax(np.absolute(result), axis = 1)
        mean_of_max = np.nanmean(max_dot_product)
        std_of_max = np.nanstd(max_dot_product, ddof = 1)
        print('The average of the maximum absolute value of dot product is ', mean_of_max)
        print('The standard deviation of the maximum absolute value of dot product is ', std_of_max)
        
        if not os.path.exists('main_program_results'):
            os.mkdir('main_program_results')
            
        # create file name if a search was performed
        if search_method != None:
            # if user is using all of catalogue, record this fact and epsilon and
            # v_scale in file name
            if search_method == "all of local":
                file_name = 'epsilon = {}, v_scale = {}, full sample'.format(
                    epsilon, v_scale)
            # if actual search was done, record search star as well
            else:
                file_name = 'epsilon = {}, v_scale = {}, star galactocentric = {}'.format(
                        epsilon, v_scale, np.array_str(point_galactocentric))
        # remove any line with \n in the title
        file_name = file_name.replace('\n','')
        # save result
        np.savez('main_program_results/' + file_name, cluster = cluster, result = result)
        
        # create graph of kmeans projection in 2 dimension
        fig = plt.figure(figsize=(8, 8))
        # only plot projection of samples in x and y dimension
        plt.scatter(samples[:,0], samples[:,1], s=15, c='blue')
        plt.scatter(cluster[:, 0], cluster[:, 1], s=30, c='red')
        plt.title("K-Means Cluster Centers in xy Dimension", fontsize=20)
        plt.xlabel('x / 8 kpc', fontsize = 15)
        plt.ylabel('y / 8 kpc', fontsize = 15)
        # save figure
        kmeans_figure_name = file_name + ', kmeans xy figure.pdf'
        plt.savefig('main_program_results/' + kmeans_figure_name)
        plt.show()
        
        # create graph of dot product
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        # get the maximum dot product at each cluster center
        # change all nan to 0 in result for graphing purpose
        result2 = np.nan_to_num(result)
        max_dot_product2 = np.max(np.absolute(result2), axis = 1)
        # scatter the cluster center x, y, and height is max dot product
        ax.scatter(cluster[:, 0], cluster[:, 1], max_dot_product2, s = 25)
        ax.set_title("Maximum Absolute Value of Dot Product in xy Dimension", fontsize=15)
        ax.set_xlabel('x / 8 kpc')
        ax.set_ylabel('y / 8 kpc')
        ax.set_zlabel('maximum dot product')
        # save figure
        dot_product_figure_name = file_name + ', max dot product figure.pdf'
        plt.savefig('main_program_results/' + dot_product_figure_name)
        plt.show()
        
    
if __name__ == "__main__":
    main(None, "local")
