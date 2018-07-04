"""
NAME:
    tools

PURPOSE:
    Contains miscellaneous tools, including frame conversion, interaction with
    user, unit conversion.
    
HISTORY:
    2018-05-31 - Written - Samuel Wong
    2018-06-19 - Added Amount of Standard Deviation Cut function - Michael Poon
"""
import numpy as np
import astropy.units as unit
from astropy.coordinates import SkyCoord, CartesianRepresentation, CartesianDifferential
from galpy.util import bovy_coords
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D

def galactic_to_galactocentric(point):
    """
    NAME:
        galactic_to_galactocentric

    PURPOSE:
        Given 6 coordinates for the position and velocity of a star in
        galactic Cartesian coordinate, convert to galactocentric Cartesian

    INPUT:
        point = numpy array ([u, v, w, U, V, W]), where:
            u =  u coordinate in kpc
            v = v coordinate in kpc
            w = w coordinate in kpc
            U = velocity in U in km/s
            V = velocity in V in km/s
            W = velocity in W in km/s

    OUTPUT:
        numpy array ([x, y, z, vx, vy, vz]), where:
            x =  x coordinate in kpc
            y = y coordinate in kpc
            z = z coordinate in kpc
            vx = velocity in x in km/s
            vy = velocity in y in km/s
            vz = velocity in z in km/s

    HISTORY:
        2018-05-30 - Written - Samuel Wong
    """
    u, v, w, U, V, W = point
    coord = SkyCoord(frame = 'galactic', representation_type = CartesianRepresentation,
                 differential_type = CartesianDifferential,
                 u = u*unit.kpc, v = v*unit.kpc, w = w*unit.kpc,
                 U = U*unit.km/unit.s, V = V*unit.km/unit.s, W = W*unit.km/unit.s)
    coord = coord.transform_to('galactocentric')
    coord.representation_type = CartesianRepresentation
    x = coord.x.value
    y = coord.y.value
    z = coord.z.value
    vx = coord.v_x.value
    vy = coord.v_y.value
    vz = coord.v_z.value
    return np.array([x, y, z, vx, vy, vz])


def galactocentric_to_galactic(point):
    """
    NAME:
        galactocentric_to_galactic

    PURPOSE:
        Given 6 coordinates for the position and velocity of a star in
        galactocentric Cartesian coordinate, convert to galactic Cartesian

    INPUT:
        point = numpy array ([x, y, z, vx, vy, vz]), where:
            x =  x coordinate in kpc
            y = y coordinate in kpc
            z = z coordinate in kpc
            vx = velocity in x in km/s
            vy = velocity in y in km/s
            vz = velocity in z in km/s

    OUTPUT:            
        numpy array ([u, v, w, U, V, W]), where:
            u =  u coordinate in kpc
            v = v coordinate in kpc
            w = w coordinate in kpc
            U = velocity in U in km/s
            V = velocity in V in km/s
            W = velocity in W in km/s

    HISTORY:
        2018-05-30 - Written - Samuel Wong
    """
    x, y, z, vx, vy, vz = point
    coord = SkyCoord(frame = 'galactocentric', representation_type = CartesianRepresentation,
                 differential_type = CartesianDifferential,
                 x = x*unit.kpc, y = y*unit.kpc, z = z*unit.kpc,
                 v_x = vx*unit.km/unit.s, v_y = vy*unit.km/unit.s,
                 v_z = vz*unit.km/unit.s)
    coord = coord.transform_to('galactic')
    coord.representation_type = CartesianRepresentation
    u = coord.u.value
    v = coord.v.value
    w = coord.w.value
    U = coord.U.value
    V = coord.V.value
    W = coord.W.value
    return np.array([u, v, w, U, V, W])

def to_natural_units(list_of_coord, ro=8., vo=220.):
    """
    NAME:
        to_natural_units

    PURPOSE:
        Given a list of coordinates, return a list of the same shape but
        changed to natural units by taking the ratio with the Sun's positio
        and velocity
        
    INPUT:
        list_of_coord = an array of coordinates, each coordinate has 6 components
            of positions and velocities, in cartesian galactocentric.
            Assumed to be in unit of kpc and km/s

    OUTPUT:            
        natural_list = an array of coordinates, each coordinate has 6 components
            of positions and velocities, in natural units
            
        ro = distance from galactic centre to the vantage point
        
        vo = circular velocity at ro

    HISTORY:
        2018-05-30 - Written - Samuel Wong
        2018-06-27 - Modified so that it is more efficient by dividing all
                     of the components at once in numpy - Samuel Wong
    """
    pos = list_of_coord[:,:3] / ro
    vel = list_of_coord[:,3:] / vo
    return np.concatenate((pos, vel), axis=1)

def to_physical_units(natural_coords, ro=8., vo=220.):
    """
    NAME:
        to_natural_units
        
    PURPOSE:
        given a list of coordinates in natural units, convert to physical units
        assuming ro=8. and vo=220.
        
    INPUT:
        natural_coords - Nx6 array of rectangular galactocentric coordinates of 
        the form (x, y, z, vx, vy, vz) in natural units
        
        ro = distance from galactic centre to the vantage point
        
        vo = circular velocity at ro
        
    OUPUT:
        Nx6 array of rectangular galactocentric coordinates of the form 
        (x, y, z, vx, vy, vz) in [kpc, kpc, kpc, km/s, km/s, km/s]
    """
    pos = natural_coords[:,:3] * ro
    vel = natural_coords[:,3:] * vo
    return np.concatenate((pos, vel), axis=1)

def get_star_coord_from_user():
    """
    NAME:
        get_star_coord_from_user

    PURPOSE:
        
    Input: 
        User is prompted to input coordinate of stars. Allowed to choose by
        default Sun.

    OUTPUT:
        (point_galactocentric, point_galactic) where each component is a 
        numpy array, which contains 6 numbers, 3 position and 3 velocity, in 
        galactocentric and galactic, respectively.
        The numbers are assumed to be in unit of kpc and km/s.

    HISTORY:
        2018-05-30 - Written - Samuel Wong
    """
    # initialize repeat boolean
    repeat = True
    
    while (repeat):
        # ask user if want to default to sun
        sun = input("Do you want to search around the Sun? (y/n) ")
        if sun == 'y' or sun == 'yes' or sun == 'Yes':
            point_galactic = np.array([0, 0, 0, 0, 0, 0])
            point_galactocentric = galactic_to_galactocentric(point_galactic)
            print('Searching around Sun.')
            repeat = False
        else:
            # ask the user for input coordinate frame
            frame = input("Do you want to search star in galactic or galactocentric coordinate? ")
            if frame == "galactic":
                repeat = False
                print("Please enter position in kpc and velocity in km/s.")
                u  = float(input('u = '))
                v  = float(input('v = '))
                w  = float(input('w = '))
                U  = float(input('U = '))
                V  = float(input('V = '))
                W  = float(input('W = '))
                point_galactic = np.array([u, v, w, U, V, W])
                point_galactocentric = galactic_to_galactocentric(point_galactic)
            elif frame == "galactocentric":
                repeat = False
                print("Please enter position in kpc and velocity in km/s.")
                x  = float(input('x = '))
                y  = float(input('y = '))
                z  = float(input('z = '))
                vx  = float(input('vx = '))
                vy = float(input('vy = '))
                vz  = float(input('vz = '))
                point_galactocentric = np.array([x, y, z, vx, vy, vz])
                point_galactic = galactocentric_to_galactic(point_galactocentric)
            
    return (point_galactocentric, point_galactic)


def create_meshgrid(xy_min, xy_max, xy_spacing, z_min, z_max, z_spacing,
                    vxy_min, vxy_max, vxy_spacing, vz_min, vz_max, vz_spacing):
    """
    NAME:
        create_meshgrid

    PURPOSE:
        Create a meshgrid of R^6 values used to evaluate many points easily along
        a given density function.

    Input:
        User is prompted to input the minimum, maximum and spacing values of each
        galactocentric coordinate (x, y, z, vx, vy, vz). The meshgrid goes from minimum
        to maximum inclusive.

        Note: xy are put together to simplify input as we assume the bounds and spacing should
        be the same. We assume the circle of Milky Way stars is symmetric in x, y, vx and vy.
        vxy_min and vxy_max refers to the bounds of vx and vy as the same.

    OUTPUT:
        numpy array of 6 dimensional tuples, in the order of (x, y, z, vx, vy, vy)
        x, y and x are in kpc.
        vx, vy and vz are in km/s.
        We artifically remove any point whose (x,y,z) = (0,0,0)

    HISTORY:
        2018-06-04 - Written - Michael Poon
    """

    x = np.arange(xy_min, xy_max + xy_spacing, xy_spacing)
    y = np.arange(xy_min, xy_max + xy_spacing, xy_spacing)
    z = np.arange(z_min, z_max + z_spacing, z_spacing)
    vx = np.arange(vxy_min, vxy_max + vxy_spacing, vxy_spacing)
    vy = np.arange(vxy_min, vxy_max + vxy_spacing, vxy_spacing)
    vz = np.arange(vz_min, vz_max + vz_spacing, vz_spacing)

    x_values, y_values, z_values, vx_values, vy_values, vz_values = np.meshgrid(x, y, z, vx, vy, vz)

    x_values = x_values.reshape(-1, 1)  # -1 is flatten and 1 is put into 1D column
    y_values = y_values.reshape(-1, 1)
    z_values = z_values.reshape(-1, 1)
    vx_values = vx_values.reshape(-1, 1)
    vy_values = vy_values.reshape(-1, 1)
    vz_values = vz_values.reshape(-1, 1)

    meshgrid = np.concatenate((x_values, y_values, z_values, vx_values, vy_values, vz_values), axis=1)
    
    return meshgrid


def std_cut(samples, number_of_std_cut):
    
    """
    NAME:
        std_cut
    
    PURPOSE:
        Remove outliers (points with any coordinate value further 
        than a multiple of std from the mean) as a quality cut.
        
    INPUT:
        1. Nx6 array of rectangular phase space coordinates of the form 
        (x, y, z, vx, vy, vz) in [kpc, kpc, kpc, km/s, km/s, km/s].
        
        2. Amount of standard deviation cut from each coordinate
        
    OUTPUT:
        Nx6 array of rectangular phase space coordinates of the form 
        (x, y, z, vx, vy, vz) in [kpc, kpc, kpc, km/s, km/s, km/s].
    
    NOTES:
        From a sample of 6376803 (20% parallax cut), if we do a cut of 
        (3.0 std) -> new_sample = 5950618 (93.3% remaining / 426185 cut)
        (2.5 std) -> new_sample = 5658852 (88.7% remaining / 717951 cut)
        (2.0 std) -> new_sample = 5131168 (80.4% remaining / 1245635 cut)
        
    HISTORY:
        2018-06-04 - Written - Michael Poon
    """
    
    print("With", len(samples), "samples, now performing a", number_of_std_cut, "std quality cut.")
    
    mean, std = np.mean(samples, axis=0), np.std(samples, axis=0, ddof=1)
    new_samples = samples[np.all(np.abs((samples - mean) / std) < number_of_std_cut, axis=1)]
    print(len(samples) - len(new_samples), "samples cut. Now there are", len(new_samples), "samples.")
    return new_samples    

def rect_to_cyl(x, y, z, vx, vy, vz):
    """
    NAME:
        rect_to_cyl
        
    PURPOSE:
        convert rectangular coordinates to cylindrical coordinates with velocity
        
    INPUT:
        x, y, z, vx, vy, vz - rectangular coordinates; can be arrays or
        individual values
        
    OUTPUT:
        R, vR, vT, z, vz, phi
    """
    R, phi, z = bovy_coords.rect_to_cyl(x, y, z)
    vR, vT, vz = bovy_coords.rect_to_cyl_vec(vx, vy, vz, x, y, z)
    
    if not isinstance(x, np.ndarray):
        return np.array((R, vR, vT, z, vz, phi)).reshape((1,-1))
    else:
        return np.stack((R, vR, vT, z, vz, phi), axis=1)

def cyl_to_rect(R, vR, vT, z, vz, phi):
    """
    NAME:
        cyl_to_rect
        
    PURPOSE:
        convert cylindrical coordinates to rectangular coordinates with velocity
        
    INPUT:
        R, vR, vT, z, vz, phi - cylindrical coordinates; can be arrays or
        individual values
        
    OUTPUT:
        x, y, z, vx, vy, vz
    """
    x, y, z = bovy_coords.cyl_to_rect(R, phi, z)
    vx, vy, vz = bovy_coords.cyl_to_rect_vec(vR, vT, vz, phi)

    if not isinstance(x, np.ndarray):
        return np.array((x, y, z, vx, vy, vz)).reshape((1,-1))
    else:
        return np.stack((x, y, z, vx, vy, vz), axis=1)


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
    # create graph of kmeans projection in 2 dimension
    fig = plt.figure(figsize=(8, 8), facecolor='black')
    plt.style.use("dark_background")
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
    
    
def color_plot_ij(max_dot_product, cluster, file_name, i, j):
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
    # filter out nan
    cluster = cluster[~np.isnan(max_dot_product)]
    max_dot_product = max_dot_product[~np.isnan(max_dot_product)]
    # create plot    
    plt.figure(figsize=(10,8), facecolor='black')
    plt.style.use("dark_background")
    # scatter plot the cluster, with the 2 given projection axis
    # we need to use the transpose to get all the components, instead of 
    # 6 components for each star.
    # set the color to the dot product
    plt.scatter(*cluster.T[[i,j]], c=max_dot_product, marker='.', s=5, 
                cmap='plasma', vmin=0, vmax=1)
    plt.colorbar(label='Maximum Absolute Dot Product')
    # get the axis name
    x_axis, x_divisor = get_axis_from_index(i)
    y_axis, y_divisor = get_axis_from_index(j)
    plt.xlabel('${}/{}$'.format(x_axis, x_divisor))
    plt.ylabel('${}/{}$'.format(y_axis, y_divisor))
    plt.title("Maximum Absolute Value of Dot Product in {}-{} Dimension".format(
            x_axis, y_axis))
    # save figure
    color_figure_name = 'color dot product {}-{} figure.png'.format(
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

def color_plot(max_dot_product, cluster, file_name):
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
    # go through al combinations of axis projection and plot them
    for i in range(6):
        for j in range(i + 1, 6):
            color_plot_ij(max_dot_product, cluster, file_name, i, j)