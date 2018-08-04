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
    # create graph of dot product
    fig = plt.figure(figsize=(8, 8), facecolor='black')
    plt.style.use("dark_background")
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
    plt.show()
    

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
    
        plt.figure(figsize=(10,8))
        plt.hist(result, bins='auto', density=True, range=(0,1))
        plt.xlabel('Fractional Length of Projection')
        plt.ylabel('Frequency')
        plt.title('Fractional Length of Projection')
        plt.savefig('main_program_results/' + file_name + 
                    '/fractional length histogram.png')
    elif uniformity_method == 'dot product':
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
        
        plt.figure(figsize=(10,8))
        plt.hist(result, bins='auto', density=True, range=(0,1))
        plt.xlabel('Maximum Absolute Dot Product')
        plt.ylabel('Frequency')
        plt.title('Maximum Absolute Values of the Dot Products')
        plt.savefig('main_program_results/' + file_name + 
                    '/dot product histogram.png')
            
    with plt.style.context(('dark_background')):
        # go through al combinations of axis projection and plot them
        for i in range(6):
            for j in range(i + 1, 6):
                color_plot_ij(result, cluster, file_name, uniformity_method, i, j)