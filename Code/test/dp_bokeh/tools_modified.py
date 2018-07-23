"""
NAME:
    tools_modified

PURPOSE:
    Used to help plot interactive dot product projections using Bokeh linked plots.
"""

import numpy as np
import astropy.units as unit
from astropy.coordinates import SkyCoord, CartesianRepresentation, CartesianDifferential
from galpy.util import bovy_coords
from bokeh.io import output_file, show
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
import pylab as plt


def color_plot_ij(source, TOOLS, i, j, all_proj, counter):
    """
    NAME:
        color_plot_ij

    PURPOSE:
        Given result of dot product and cluster, and the index of the two
        projection axis, plot the scatter plot of the cluster, with the 
        color of the point corresponding to maximum dot product value.
        Save graph in appropriate folder.

    INPUT:
        max_dot_product = a numpy array storing the maximum dot product at each
                          cluster center
        cluster = a numpy array storing cluster centers
        file_name = a string

    OUTPUT:
        None

    HISTORY:
        2018-07-03 - Written - Mathew Bubb, Samuel Wong
    """
    
    # create plot    
    proj = figure(tools=TOOLS, plot_width=300, plot_height=300, title='lulz')
    # scatter plot the cluster, with the 2 given projection axis
    # we need to use the transpose to get all the components, instead of 
    # 6 components for each star.
    # set the color to the dot product
    x_axis, x_divisor = get_axis_from_index(i)
    y_axis, y_divisor = get_axis_from_index(j)
    
    proj.scatter('{}'.format(x_axis), '{}'.format(y_axis), color='colours', size=0.5, source=source)
    
    # get the axis name
    proj.xaxis.axis_label = '{}/{}'.format(x_axis, x_divisor)
    proj.xgrid.grid_line_alpha = 0.2
    proj.yaxis.axis_label = '{}/{}'.format(y_axis, y_divisor)
    proj.ygrid.grid_line_alpha = 0.2
    proj.background_fill_color = "black"
    #plt.title("Maximum Absolute Value of Dot Product in {}-{} Dimension".format(x_axis, y_axis))
    # save figure
    #color_figure_name = 'color dot product {}-{} figure.png'.format(x_axis, y_axis)
    #plt.savefig('main_program_results/' + file_name +'/'+ color_figure_name)
    all_proj[counter] = proj
    

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

def color_plot(source, TOOLS):
    """
    NAME:
        color_plot

    PURPOSE:
        Given result of dot product and cluster, plot all possible 2
        dimensional projection scatter plot with color corresponding to dot 
        product values. Save all the graph in the corresponding folder.

    INPUT:
        max_dot_product = a numpy array storing the maximum dot product at each
                          cluster center
        cluster = a numpy array storing cluster centers
        file_name = a string

    OUTPUT:
        None

    HISTORY:
        2018-07-03 - Written - Samuel Wong
    """
    # go through all combinations of axis projection and plot them
    counter=0
    all_proj = [None] * 15
    for i in range(6):
        for j in range(i + 1, 6):
            color_plot_ij(source, TOOLS, i, j, all_proj, counter)
            counter+=1
    show(gridplot([[all_proj[0], None,        all_proj[2], all_proj[6], all_proj[9]], 
                   [all_proj[1], all_proj[5], all_proj[3], all_proj[7], all_proj[10]],
                   [all_proj[12], None,       all_proj[4], all_proj[8], all_proj[11]],
                   [all_proj[13], all_proj[14], None,     None,        None]]))