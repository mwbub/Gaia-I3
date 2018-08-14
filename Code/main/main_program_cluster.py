"""
NAME:
    main_program_cluster
PURPOSE:
    Evaluate uniformity of dot product on a cluster of points in phase space
    given by kmeans.
    
HISTORY:
    2018-06-20 - Written - Samuel Wong
    2018-07-20 - Changed to analytic gradient - Samuel Wong
    2018-07-26 - Changed everything to using arrays and added projection
                 method - Samuel Wong
    2018-08-14 - Added option to divide by selection in density - Samuel Wong
"""
import time as time_class
import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import os, sys
import dill
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
from tools.plots import *
#get standard selection function
with open("../selection/parallax selection function/selection_function", "rb") as dill_file:
    standard_selection = dill.load(dill_file)

# create a subfolder to save results
if not os.path.exists('main_program_results'):
    os.mkdir('main_program_results')


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
    if search_method != "all of local":
        epsilon = float(input("epsilon = "))
        v_scale = float(input("v_scale = "))
        point_galactocentric, point_galactic = get_star_coord_from_user()
    
    if search_method == "online":
        samples = search_online.search_phase_space(*point_galactic, epsilon, v_scale)
    elif search_method == "local":
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
        file_name = ('epsilon = {}, v_scale = {}, star galactocentric = '
                     '[{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}]'
                     ).format(epsilon, v_scale, *point_galactocentric)
    # remove any line with \n in the title
    file_name = file_name.replace('\n','')
    return samples, file_name


def get_samples_density_filename(custom_density, search_method, custom_samples,
                                 uniformity_method, selection):
    """
    NAME:
        get_samples_density_filename
    PURPOSE:
        Given the input of the main functions, which are custom density,
        search method, and custom samples, check whether custom density or 
        custom samples are None. If they are, search for stars in Gaia and use
        KDE to generate density. If one of them is not None, use the custom
        density or custom samples. Also, generate appropriate file name and 
        create a subfolder to save file.
    INPUT:
        custom_density = a function that takes 6 values in an array and outputs
                         the density at that point; can also be None
        search_method = one of "online", "local", "all of local"
        custom_samples = an N by 6 arrays in physical units representing 6
                         dimensional star coordinate; can also be None
        uniformity_method = "projection" or "dot product", referring to how
                            uniformity of density is evaluated
        selection = a selection function that takes parallax to Sun and returns
                    fraction of stars that are left after selection;
                    takes array; takes parallax in physical units
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
        density = generate_KDE(samples, 'epanechnikov', selection)
    
    # add presence of selection in filename
    if selection is not None:
        file_name = file_name + ' with selection'
    # create a sub-folder to save results wihout further specification of 
    # uniformity method
    if not os.path.exists('main_program_results/'+file_name):
        os.mkdir('main_program_results/' + file_name)
    # add uniformity sub folder to file name
    file_name = file_name + '/' + uniformity_method + '/'
    # add a directory a level deeper
    if not os.path.exists('main_program_results/'+file_name):
        os.mkdir('main_program_results/' + file_name)
        
    return samples, density, file_name


def get_cluster(samples, custom_centres):
    """
    NAME:
        get_cluster
    PURPOSE:
        Given a sample, use kmeans minibatch to find some cluster centers that
        are a good representative of the samples.
    INPUT:
        samples = a numpy arrays containing 6 dimensional coordinates in
                  galactocentric Cartesian form with natural units
                  
        custom_centres = a custom array of cluster centres at which to evaluate
                 uniformity; if None, will use kmeans clustering to get
                 the cluster centres
    OUTPUT:
        cluster = a numpy arrays containing 6 dimensional coordinates in
                  galactocentric Cartesian form with natural units; a smal but 
                  representative cluster centers of samples
    HISTORY:
        2018-06-25 - Written - Samuel Wong
    """
    if custom_centres is not None:
        cluster = custom_centres
    else:
        # let batch size be 1% of the number of samples
        batch_size = int(0.01 * np.shape(samples)[0])
        # let the number of cluster centers to be 0.1% of number of samples
        cluster_number = int(0.001 * np.shape(samples)[0])
        # use kmenas to generate a cluster of points
        cluster = kmeans(samples, cluster_number, batch_size)
    return cluster


def get_Energy_Lz_gradient(cluster, gradient_method, custom_potential):
    if gradient_method == "analytic":
        Energy_gradient = del_E(cluster, custom_potential)
        Lz_gradient = del_Lz(cluster)
    elif gradient_method == "numeric":
        Energy_gradient = grad_multi(
                lambda coord: Energy(coord, custom_potential), cluster)
        Lz_gradient = grad_multi(L_z, cluster)
    return Energy_gradient, Lz_gradient


def summary_save(result, cluster, file_name, uniformity_method):
    if uniformity_method == "dot product":
        max_dot_product = np.nanmax(np.absolute(result), axis = 1)
        mean_of_max = np.nanmean(max_dot_product)
        std_of_max = np.nanstd(max_dot_product, ddof = 1)
        print('The average of the maximum absolute value of dot product is ',
              mean_of_max)
        print('The standard deviation of the maximum absolute value \
              of dot product is ', std_of_max)
        np.savez('main_program_results/' + file_name +'/'+ 'data', 
                 cluster = cluster, result = result)
    elif uniformity_method == "projection":
        mean_projection = np.nanmean(result)
        std_projection = np.nanstd(result, ddof = 1)
        print('The average of the projection is ', mean_projection)
        print('The standard deviation of the projection is ', std_projection)
        np.savez('main_program_results/' + file_name +'/'+ 'projection data', 
                 cluster = cluster, result = result)


def main(uniformity_method = "projection", gradient_method = "analytic",
         search_method = "local", custom_density = None, custom_samples = None,
         custom_centres = None, custom_potential = None,
         selection = standard_selection):
    """
    NAME:
        main
    PURPOSE:
        Call on all modules to evaluate uniformity of density on a cluster of 
        points provided by kmeans (or using custom centres). Allows user to 
        choose between using dot product or projection method to evaluate
        uniformity. Allows the user to specify search method to to get the data
        from Gaia (either locally or online; and either everything, or an
        epsillon ball around Sun). Also, allows user to give custom density
        function or custom samples. Output figures of results, including kmeans
        and dot product scatter plot.
    INPUT:
        uniformity_method = "projection" or "dot product", referring to how
                            uniformity of density is evaluated
        gradient_method = "analytic" or "numeric", referring to how gradient
                            of energy and L_z are generated
        search_method = search the gaia catalogue online ("online"),
                        locally on a downloaded file ('local'), or use the
                        the entire downloaded gaia rv file ('all of local')
        custom_density = a customized density functiont that takes an array
                         of 6 numbers representing the coordinate and return
                         the density; if this input is None, then the code
                         will use a search method to get data from Gaia catalogue
                         and use KDE to genereate a density function
        custom_samples = an N by 6 array that represents the custom samples, 
                        with each component representing (x,y,z,vx,vy,vz),
                        respectively. They are in physical units.
        custom_centres = a custom array of cluster centres at which to evaluate
                         uniformity; if None, will use kmeans clustering to get
                         the cluster centres
        custom_potential = a galpy potential object; or None, which defaults
                        MWPotential2014
        selection = a selection function that takes parallax to Sun and returns
                    fraction of stars that are left after selection;
                    takes array; takes parallax in physical units
    HISTORY:
        2018-06-20 - Written - Samuel Wong
        2018-06-21 - Added option of custom samples - Samuel Wong and Michael
                                                      Poon
        2018-06-22 - Added Figure - Samuel Wong
        2018-07-15 - Added choice of gradient method - Samuel Wong
        2018-07-31 - Added choice of custom potential - Samuel Wong
    """        
    samples, density, file_name = get_samples_density_filename(
            custom_density, search_method, custom_samples, uniformity_method,
            selection)
    
    cluster = get_cluster(samples, custom_centres)
    
    Energy_gradient, Lz_gradient = get_Energy_Lz_gradient(
            cluster, gradient_method, custom_potential)
        
    start = time_class.time()
    result = evaluate_uniformity(density, cluster, Energy_gradient,
                                 Lz_gradient, uniformity_method)
    inter_time = time_class.time() - start
    print('time per star =', inter_time/np.shape(cluster)[0])
    
    summary_save(result, cluster, file_name, uniformity_method)        
    kmeans_plot(samples, cluster, file_name)
    color_plot(result, cluster, file_name, uniformity_method, custom_potential)
    color_plot_bokeh(result, cluster, file_name, uniformity_method)
    errorbar_plot(result, cluster, file_name, uniformity_method, 
                  custom_potential)
       
  
if __name__ == "__main__":
    main(uniformity_method = "projection", gradient_method = "analytic",
         search_method = "local", custom_density = None, custom_samples = None,
         custom_centres = None, custom_potential = None,
         selection = standard_selection)
    