"""
NAME:
    bokeh_select_range

PURPOSE:
    Used to help plot interactive dot product and fractional length projections using Bokeh linked plots.
    This is different from bokeh in main program because we CAN CHOOSE A RANGE
    
HOW TO USE:
    
"""

import numpy as np
import astropy.units as unit
from astropy.coordinates import SkyCoord, CartesianRepresentation, CartesianDifferential
from galpy.util import bovy_coords
from bokeh.io import output_file, show
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, ColorBar, LinearColorMapper, CustomJS, Button, Div
from bokeh.plotting import figure
from bokeh.models.annotations import Title
import matplotlib as mpl
import pylab as plt
plt.ioff() # turn off plots


def color_plot_ij_bokeh(source, TOOLS, i, j, all_proj, counter, uniformity_method):
    """
    NAME:
        color_plot_ij

    PURPOSE:
        Plots in either dot product or fractional length projections

    INPUT:
        source = dictionary-like data struction in Bokeh to update selected values in real-time
        TOOLS = list of tools used in bokeh plots
        i, j = projection axes
        counter = used to order plots in position/velocity space when outputted
        uniformity method = dot product/projection, projection refers to fractional length

    OUTPUT:
        None

    HISTORY:
        2018-07-30 - Written - Michael Poon
    """
    
    # get axes 
    x_axis, x_divisor = get_axis_from_index(i)
    y_axis, y_divisor = get_axis_from_index(j)
    
    
    # create plot    
    proj = figure(tools=TOOLS, plot_width=350, plot_height=300)
    proj.scatter('{}'.format(x_axis), '{}'.format(y_axis), color='colours', size=0.5, fill_alpha=1, line_alpha=1, alpha=1, source=source)
    proj.outline_line_color = None
    proj.border_fill_color = None
    proj.xgrid.grid_line_alpha = 0.2
    proj.ygrid.grid_line_alpha = 0.2
    color_mapper = LinearColorMapper(palette="Plasma256", low=0, high=1)
    color_bar = ColorBar(color_mapper=color_mapper, label_standoff=8, width=15, location=(0,0))
    proj.add_layout(color_bar, 'right')
    proj.xaxis.axis_label = '{}/{}'.format(x_axis, x_divisor)
    proj.yaxis.axis_label = '{}/{}'.format(y_axis, y_divisor)
    proj.background_fill_color = "black"
    
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

def color_plot_bokeh(source, TOOLS, uniformity_method):
    """
    NAME:
        color_plot

    PURPOSE:
        Given result of dot product and cluster, plot all possible 2
        dimensional projection scatter plot with color corresponding to dot 
        product values. Save all the graph in the corresponding folder.

    INPUT:
        source = dictionary-like data struction in Bokeh to update selected values in real-time
        TOOLS = list of tools used in bokeh plots
        uniformity method = dot product/projection, projection refers to fractional length

    OUTPUT:
        None

    HISTORY:
        2018-07-30 - Written - Michael Poon
    """
    
    # Go through all combinations of axis projection and plot them
    counter=0
    all_proj = [None] * 15
    for i in range(6):
        for j in range(i + 1, 6):
            color_plot_ij_bokeh(source, TOOLS, i, j, all_proj, counter, uniformity_method)
            counter+=1
    
    # Adding titles and adjusting graph spacing
    t1 = Title()
    t1.text = 'Position Space Projections'
    all_proj[0].title = t1
    t2 = Title()
    t2.text = 'Velocity Space Projections'
    all_proj[12].title = t2
    t3 = Title()
    t3.text = 'Position-Velocity Space Projections'
    all_proj[6].title = t3
    t4 = Title()
    if uniformity_method == "dot product":
        t4.text = "Maximum Dot Product"
    else:
        t4.text = "Fractional Length of Proj."
    all_proj[2].title = t4
    all_proj[2].title.text_font_size = "14pt"
    
    # add button to show average dot product/projection
    if uniformity_method == "dot product":
        button = Button(label='Dot Product Method', button_type='warning', width=350)
    else:
        button = Button(label='Projection Method', button_type='warning', width=350)
        
    button.callback = CustomJS(args=dict(source=source), 
                               code="""
                               var inds = source.selected['1d'].indices;
                               var data = source.data;
                               var selected_dp = 0.;
                               var total_dp = 0.;
                               
                               if (inds.length != 0) {
                                   for (i = 0; i < inds.length; i++) {
                                        selected_dp += data['result'][inds[i]];
                                   }     
                                   window.alert(inds.length + " of " + data['result'].length + " points chosen. Average Result = " + (selected_dp/inds.length).toFixed(2));
                               }
                                   
                               else {
                                   for (i = 0; i < data['result'].length; i++) {
                                        total_dp += data['result'][i];
                                   } 
                                   window.alert("No points chosen. Average Result (of all points) = " + (total_dp/data['result'].length).toFixed(2));
                               }
                          
                               """)
    
    show(gridplot([[all_proj[0],  button,       all_proj[2], all_proj[6], all_proj[9]], 
                   [all_proj[1],  all_proj[5],  all_proj[3], all_proj[7], all_proj[10]],
                   [all_proj[12], None,         all_proj[4], all_proj[8], all_proj[11]],
                   [all_proj[13], all_proj[14], None,        None,        None]]))
        
    # Show twice to compare when selecting regions    
    show(gridplot([[all_proj[0],  button,       all_proj[2], all_proj[6], all_proj[9]], 
                   [all_proj[1],  all_proj[5],  all_proj[3], all_proj[7], all_proj[10]],
                   [all_proj[12], None,         all_proj[4], all_proj[8], all_proj[11]],
                   [all_proj[13], all_proj[14], None,        None,        None]]))
    
if __name__ == "__main__":
    
    # User Input
    uniformity_method = "projection" #projection or dot product
    MIN_VAL=0.
    MAX_VAL=1.
    output_file("{}.html".format(uniformity_method))
    data = np.load('projection data.npz')
            
    # Set up data
    cluster = data['cluster']
    result = data['result']
    
    if uniformity_method == "dot product":
        result = np.nanmax(np.absolute(result), axis = 1)
    
    # Full range
    cluster = cluster[~np.isnan(result)]
    result = result[~np.isnan(result)]
    colours = ["#%02x%02x%02x" % (int(r), int(g), int(b)) for r, g, b, _ in 255*mpl.cm.plasma(mpl.colors.Normalize()(result))]
    
    # Filter by a specified range of dot product
    
    if MIN_VAL != 0. or MIN_VAL != 1.:
        cluster = cluster[(MIN_VAL < result) & (result < MAX_VAL)]
        result_temp = result[(MIN_VAL < result) & (result < MAX_VAL)]
        
        # Covert list to array
        colours = np.array([[colours]])
        colours = colours.flatten()
        
        colours = colours[(MIN_VAL < result) & (result < MAX_VAL)]
        result = result_temp

          
    # create a column data source for the plots to share
    source = ColumnDataSource(data=dict(x=cluster.T[0], y=cluster.T[1], z=cluster.T[2],
                                        vx=cluster.T[3], vy=cluster.T[4], vz=cluster.T[5], result=result, colours=colours))
    
    #denote interactive tools for linked plots
    TOOLS = "lasso_select,box_select"
    
    # create and save graph of dot product in color scatter plot in all 
    # 2 dimensional projections.
    color_plot_bokeh(source, TOOLS, uniformity_method)
        
    
        