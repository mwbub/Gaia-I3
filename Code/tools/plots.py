import sys
sys.path.append('../check_uniformity_of_density')

import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from Integral_of_Motion import Energy, L_z

def kmeans_plot(samples, cluster, file_name):
    """
    NAME:
        kmeans_plot

    PURPOSE:
        Given samples and cluster, plot them. Given file name, save the image.

    INPUT:
        samples = a numpy array storing samples
        cluster = a numpy array storing cluster centers
        file_name = a string

    OUTPUT:
        None

    HISTORY:
        2018-06-25 - Written - Samuel Wong
    """
    with plt.style.context(('dark_background')):
        # create graph of kmeans projection in 2 dimension
        fig = plt.figure(figsize=(8, 8), facecolor='black')
        # only plot projection of samples in x and y dimension
        plt.scatter(samples[:,0], samples[:,1], s=1, c='blue')
        plt.scatter(cluster[:, 0], cluster[:, 1], s=1, c='red')
        plt.title("K-Means Cluster Centers in xy Dimension", fontsize=20)
        plt.xlabel('$x/R_0$', fontsize = 15)
        plt.ylabel('$y/R_0$', fontsize = 15)
        # save figure
        kmeans_figure_name = 'kmeans xy figure.png'
        plt.savefig('main_program_results/' + file_name +'/'+ kmeans_figure_name)
        plt.show()
    

def dot_product_plot(max_dot_product, cluster, file_name):
    """
    NAME:
        dot_product_plot

    PURPOSE:
        Given result of dot product and cluster, plot them. Given file name,
        save the image.

    INPUT:
        max_dot_product = a numpy array storing the maximum dot product at each
                          cluster center
        cluster = a numpy array storing cluster centers
        file_name = a string

    OUTPUT:
        None

    HISTORY:
        2018-06-25 - Written - Samuel Wong
    """
    with plt.style.context(('dark_background')):
        # create graph of dot product
        fig = plt.figure(figsize=(8, 8), facecolor='black')
        ax = fig.add_subplot(111, projection='3d')
        # get the maximum dot product at each cluster center
        # filter out nan
        cluster = cluster[~np.isnan(max_dot_product)]
        max_dot_product = max_dot_product[~np.isnan(max_dot_product)]
        # scatter the cluster center x, y, and height is max dot product
        ax.scatter(cluster[:, 0], cluster[:, 1], max_dot_product, s = 10)
        ax.set_title("Maximum Absolute Value of Dot Product in xy Dimension", fontsize=15)
        ax.set_xlabel('$x/R_0$')
        ax.set_ylabel('$y/R_0$')
        ax.set_zlabel('maximum dot product')
        # force the z limit to 0 and 1
        ax.set_zlim(0, 1)
        # save figure
        dot_product_figure_name = 'max dot product figure.png'
        plt.savefig('main_program_results/' + file_name +'/'+ dot_product_figure_name)
        plt.show()
    
    
def color_plot_ij(result, cluster, file_name, uniformity_method, i, j):
    """
    NAME:
        color_plot_ij

    PURPOSE:
        Given result and cluster, and the index of the two
        projection axis, plot the scatter plot of the cluster, with the 
        color of the point corresponding to result value. Title depends on
        the uniformity method.
        Save graph in appropriate folder.

    INPUT:
        result = a numpy array storing the result at each cluster center
        cluster = a numpy array storing cluster centers
        file_name = a string
        uniformity_method = "projection" or "dot product"

    OUTPUT:
        None

    HISTORY:
        2018-07-03 - Written - Mathew Bubb, Samuel Wong
        2018-07-23 - Added uniformity method - Samuel Wong
    """
    with plt.style.context(('dark_background')):
        # create plot    
        plt.figure(figsize=(10,8), facecolor='black')
        # scatter plot the cluster, with the 2 given projection axis
        # we need to use the transpose to get all the components, instead of 
        # 6 components for each star.
        # set the color to the result
        plt.scatter(*cluster.T[[i,j]], c=result, marker='.', s=5, 
                    cmap='plasma', vmin=0, vmax=1)
        if uniformity_method == "dot product":
            plt.colorbar(label='Maximum Absolute Dot Product')
        elif uniformity_method == "projection":
            plt.colorbar(label='Fractional Length of Projection')
        # get the axis name
        x_axis, x_divisor = get_axis_from_index(i)
        y_axis, y_divisor = get_axis_from_index(j)
        plt.xlabel('${}/{}$'.format(x_axis, x_divisor))
        plt.ylabel('${}/{}$'.format(y_axis, y_divisor))
        if uniformity_method == "dot product":
            plt.title("Maximum Absolute Value of Dot Product in {}-{} Dimension".format(
            x_axis, y_axis))
        elif uniformity_method == "projection":
            plt.title("Fractional Length of Projection in {}-{} Dimension".format(
            x_axis, y_axis))
    
        # save figure
        if uniformity_method == "dot product":
            color_figure_name = 'color dot product {}-{} figure.png'.format(
                    x_axis, y_axis)
        elif uniformity_method == "projection":
            color_figure_name = 'color projection {}-{} figure.png'.format(
                    x_axis, y_axis)
    
        plt.savefig('main_program_results/' + file_name +'/'+ color_figure_name)
    

def get_axis_from_index(i):
    """
    NAME:
        get_axis_from_index

    PURPOSE:
        Since there are 6 dimensions corresponding to x, y, z, vx, vy, vz,
        this function takes an index from 0 to 5, and output a string
        corresponding to the appropriate name of axis as well as a string
        that represents either solar radius or solar velocity, as corresponding
        to the axis name.

    INPUT:
        i = an integer from 0 to 5 inclusive

    OUTPUT:
        (axis_name, divisor)

    HISTORY:
        2018-07-03 - Written - Samuel Wong
    """
    # create an list storing axis name in corresponding position
    axis = [['x', 'R_0'], ['y', 'R_0'], ['z', 'R_0'], ['vx', 'v_0'],
            ['vy', 'v_0'], ['vz', 'v_0']]
    return axis[i]

def color_plot(result, cluster, file_name, uniformity_method, 
               custom_potential = None):
    """
    NAME:
        color_plot

    PURPOSE:
        Given result and cluster, plot all possible 2 dimensional projection
        scatter plot with color corresponding to result values. Save all the 
        graph in the corresponding folder.

    INPUT:
        result = a numpy array storing the result at each cluster center
        cluster = a numpy array storing cluster centers
        file_name = a string
        uniformity_method = "projection" or "dot product"
        custom_potential = potential used to evaluate energy; default = 
                           MWPotential2014

    OUTPUT:
        None

    HISTORY:
        2018-07-03 - Written - Samuel Wong
        2018-07-23 - Added uniformity method - Samuel Wong
    """
    # only work with the largest dot product, if that it is the input
    if uniformity_method == "dot product":
        result = np.nanmax(np.absolute(result), axis = 1)
    
    # filter out nan
    cluster = cluster[~np.isnan(result)]
    result = result[~np.isnan(result)]
        
    energy = Energy(cluster, custom_potential)
    angular_momentum = L_z(cluster)
            
    if uniformity_method == 'projection':
        plt.figure(figsize=(10,8))
        plt.hist(result, bins='auto', density=True, range=(0,1))
        plt.xlabel('Fractional Length of Projection')
        plt.ylabel('Frequency')
        plt.title('Fractional Length of Projection')
        plt.savefig('main_program_results/' + file_name + 
                    '/fractional length histogram.png')
        
        with plt.style.context(('dark_background')):
            plt.figure(figsize=(10,8), facecolor='black')
            plt.scatter(angular_momentum, energy, c=result, marker='.', s=5,
                        cmap='plasma', vmin=0, vmax=1)
            plt.colorbar(label='Fractional Length of Projection')
            plt.xlabel('$L_z$')
            plt.ylabel('$E$')
            plt.title('Fractional Length of Projection in $L_z-E$ Dimension')
            plt.savefig('main_program_results/' + file_name + 
                        '/color projection L_z-E figure.png')
    elif uniformity_method == 'dot product':
        plt.figure(figsize=(10,8))
        plt.hist(result, bins='auto', density=True, range=(0,1))
        plt.xlabel('Maximum Absolute Dot Product')
        plt.ylabel('Frequency')
        plt.title('Maximum Absolute Values of the Dot Products')
        plt.savefig('main_program_results/' + file_name + 
                    '/dot product histogram.png')
        
        with plt.style.context(('dark_background')):
            plt.figure(figsize=(10,8), facecolor='black')
            plt.scatter(angular_momentum, energy, c=result, marker='.', s=5,
                        cmap='plasma', vmin=0, vmax=1)
            plt.colorbar(label='Maximum Absolute Dot Product')
            plt.xlabel('$L_z$')
            plt.ylabel('$E$')
            plt.title('Maximum Absolute Value of Dot Product in $L_z-E$ Dimension')
            plt.savefig('main_program_results/' + file_name + 
                        '/color dot product L_z-E figure.png')
            
    # go through al combinations of axis projection and plot them
    for i in range(6):
        for j in range(i + 1, 6):
            color_plot_ij(result, cluster, file_name, uniformity_method, i, j)

def error_plot_ij(errors, cluster, file_name, uniformity_method, i, j):
    """
    NAME:
        error_plot_ij
        
    PURPOSE:
        Create a scatter plot of the uncertainties of the results of a run of
        the main program along the ith and jth axes in cartesian coordinates.
        
    INPUT:
        errors - the uncertainty in the result value at each cluster centre
        
        cluster - the cluster centres
        
        file_name - the name of the folder in which the plot is to be saved
        
        uniformity_method - the method used to compute the results; can be 
        "projection" or "dot product"
        
        i, j - the indices of the two axes to plot; the order of the possible
        axes is (x, y, z, vx, vy, vz)
        
    OUTPUT:
        None
    """
    with plt.style.context(('dark_background')):
        plt.figure(figsize=(10,8), facecolor='black')
        plt.scatter(*cluster.T[[i, j]], c=errors, marker='.', s=5, cmap='jet', 
                    vmin=0)
        
        x_axis, x_divisor = get_axis_from_index(i)
        y_axis, y_divisor = get_axis_from_index(j)
        plt.xlabel('${}/{}$'.format(x_axis, x_divisor))
        plt.ylabel('${}/{}$'.format(y_axis, y_divisor))
        
        if uniformity_method == 'projection':
            plt.colorbar(label='Fractional Length Uncertainty')
            plt.title('Fractional Length Uncertainties in the {}-{} '
                      'Dimension'.format(x_axis, y_axis))
            figure_name = 'projection uncertainties {}-{} figure.png'.format(
                    x_axis, y_axis)
        elif uniformity_method == 'dot product':
            plt.colorbar(label='Maximum Dot Product Uncertainty')
            plt.title('Maximum Dot Product Uncertainties in the {}-{} '
                      'Dimension'.format(x_axis, y_axis))
            figure_name = 'dot product uncertainties {}-{} figure.png'.format(
                    x_axis, y_axis)
            
        plt.savefig('main_program_results/' + file_name + figure_name)
    
def error_plot(errors, cluster, file_name, uniformity_method, 
               custom_potential = None):
    """
    NAME:
        error_plot
        
    PURPOSE:
        Plot the uncertainties of the results of a run of the main program in
        all possible 2D projections of cartesian phase space.
        
    INPUT:
        errors - the uncertainty in the result value at each cluster centre
        
        cluster - the cluster centres
        
        file_name - the name of the folder in which the plot is to be saved
        
        uniformity_method - the method used to compute the results; can be 
        "projection" or "dot product"
        
        custom_potential - galpy Potential or list of Potentials used to 
        evaluate energy; default = MWPotential2014
        
    OUTPUT:
        None
    """
    cluster = cluster[~np.isnan(errors)]
    errors = errors[~np.isnan(errors)]
        
    energy = Energy(cluster, custom_potential)
    angular_momentum = L_z(cluster)
    
    if uniformity_method == 'projection':
        with plt.style.context(('dark_background')):
            plt.figure(figsize=(10,8), facecolor='black')
            plt.scatter(angular_momentum, energy, c=errors, marker='.', s=5,
                        cmap='jet', vmin=0)
            plt.colorbar(label='Fractional Length Uncertainty')
            plt.xlabel('$L_z$')
            plt.ylabel('$E$')
            plt.title('Fractional Length Uncertainties in the $L_z-E$ '
                      'Dimension')
            plt.savefig('main_program_results/' + file_name +
                        'projection uncertainties L_z-E figure.png')
    elif uniformity_method == 'dot product':
        with plt.style.context(('dark_background')):
            plt.figure(figsize=(10,8), facecolor='black')
            plt.scatter(angular_momentum, energy, c=errors, marker='.', s=5,
                        cmap='jet', vmin=0)
            plt.colorbar(label='Maximum Dot Product Uncertainty')
            plt.xlabel('$L_z$')
            plt.ylabel('$E$')
            plt.title('Maximum Dot Product Uncertainties in the $L_z-E$ '
                      'Dimension')
            plt.savefig('main_program_results/' + file_name +
                        'dot product uncertainties L_z-E figure.png')
            
    for i in range(6):
        for j in range(i + 1, 6):
            error_plot_ij(errors, cluster, file_name, uniformity_method, i, j)
            
def errorbar_plot(result, cluster, file_name, uniformity_method, 
            custom_potential = None, errors = None):
    """
    NAME:
        1d_plot
        
    PURPOSE:
        Plot 1d scatter plots of the results of a run of the main program in 
        each dimension.
        
    INPUT:
        result - a numpy array storing the result at each cluster center
        
        cluster - the cluster centres
        
        file_name - the name of the folder in which the plot is to be saved
        
        uniformity_method - the method used to compute the results; can be 
        "projection" or "dot product"
        
        custom_potential - galpy Potential or list of Potentials used to 
        evaluate energy; default = MWPotential2014
        
        errors - a numpy array of errors in the same shape as result; if
        provided, error bars will be plotted for each point
        
    OUTPUT:
        None
    """
    axis = [['x', 'R_0'], ['y', 'R_0'], ['z', 'R_0'], ['vx', 'v_0'],
            ['vy', 'v_0'], ['vz', 'v_0']]
    
    # convert dot products to max dot products if not done already
    if uniformity_method == 'dot product' and result.ndim != 1:
        result = np.nanmax(np.abs(result), axis = 1)
    
    # filter out nans in errors
    if errors is not None:
        cluster = cluster[~np.isnan(errors)]
        result = result[~np.isnan(errors)]
        errors = errors[~np.isnan(errors)]
        errors = errors[~np.isnan(result)]
    
    # filter out nans in result
    cluster = cluster[~np.isnan(result)]
    result = result[~np.isnan(result)]
        
    energy = Energy(cluster, custom_potential)
    angular_momentum = L_z(cluster)
    
    for i in range(len(axis)):
        plt.figure(figsize=(10,8))
        plt.errorbar(cluster[:,i], result, yerr=errors, fmt='.', 
                     ecolor='lightgrey')
        plt.ylim((0,1))
        plt.xlabel('${}/{}$'.format(*axis[i]))
        
        if uniformity_method == 'projection':
            plt.ylabel('Fractional Length of Projection')
            plt.title('Fractional Length of Projection in the ${}$ '
                      'Dimension'.format(axis[i][0]))
            plt.savefig('main_program_results/' + file_name +
                        'projection {} figure.png'.format(axis[i][0]))
        elif uniformity_method == 'dot product':
            plt.ylabel('Maximum Absolute Dot Product')
            plt.title('Maximum Absolute Dot Product in the ${}$ '
                      'Dimension'.format(axis[i][0]))
            plt.savefig('main_program_results/' + file_name +
                        'dot product {} figure.png'.format(axis[i][0]))
            
    plt.figure(figsize=(10,8))
    plt.errorbar(energy, result, yerr=errors, fmt='.', ecolor='lightgrey')
    plt.ylim((0,1))
    plt.xlabel('$E$')
    
    if uniformity_method == 'projection':
        plt.ylabel('Fractional Length of Projection')
        plt.title('Fractional Length of Projection in the $E$ Dimension')
        plt.savefig('main_program_results/' + file_name + 
                    'projection E figure.png')
    elif uniformity_method == 'dot product':
        plt.ylabel('Maximum Absolute Dot Product')
        plt.title('Maximum Absolute Dot Product in the $E$ Dimension')
        plt.savefig('main_program_results/' + file_name +
                    'dot product E figure.png')
        
    plt.figure(figsize=(10,8))
    plt.errorbar(angular_momentum, result, yerr=errors, fmt='.', 
                 ecolor='lightgrey')
    plt.ylim((0,1))
    plt.xlabel('$L_z$')
    
    if uniformity_method == 'projection':
        plt.ylabel('Fractional Length of Projection')
        plt.title('Fractional Length of Projection in the $L_z$ Dimension')
        plt.savefig('main_program_results/' + file_name +
                    'projection L_z figure.png')
    elif uniformity_method == 'dot product':
        plt.ylabel('Maximum Absolute Dot Product')
        plt.title('Maximum Absolute Dot Product in the $L_z$ '
                  'Dimension')
        plt.savefig('main_program_results/' + file_name +
                    'dot product L_z figure.png')
        