"""
NAME:
    sampleV_on_set
PURPOSE:
    Sample velocity of a set of points in a density function by using
    interpolation of a grid, thereby improving the efficiency of sampleV.
    
HISTORY:
    2018-06-11 - Written - Samuel Wong
"""
import numpy as np
from scipy.interpolate import RectBivariateSpline as interpolation


def sampleV_on_set(rz_set, df):
    """
    NAME:
        sampleV_on_set

    PURPOSE:
        Given a three dimensional density function (df), as well as a set of 
        r and z coordinates of stars, return three sampled velocity for each
        star.

    INPUT:
        rz_set = a numpy array containing a list of (r,z) coordinate; assumed
                 to be all in natural unit
        df = a galpy three dimensional density function

    OUTPUT:
        coord_v = a numpy array containing the original coordinate but
                        with velocity attached. Each coordinate is of the form
                        (r, z, vR, vT, vz)

    HISTORY:
        2018-06-11 - Written - Samuel Wong
    """
    # separate the coodinates into outliers and normal points.
    # outliers are defined to be values more than 3 standard deviation
    normal, outliers = separate_outliers(rz_set, 3)
    
    # initialize numpy array storing result of outliers
    outlier_coord_v = np.empty((outliers.shape[0], 5))
    # sample the velocity of outliers directly
    for i, outlier in enumerate(outliers):
        R, z = outlier
        vR, vT, vz = df.sampleV(R, z)
        outlier_coord_v[i] = np.array([R, z, vR, vT, vz])
    
    # for the normal stars, we will be evaluating sample v on a grid and doing
    # interpolation on it
    # initialize numpy array storing result of normal points
    normal_coord_v = np.empty((normal.shape[0], 5))
    # optimize the dimensions of the grid
    R_number, z_number = optimize_grid_dim(normal)
    # get grid
    grid, R_linspace, z_linspace = generate_grid(normal, R_number, z_number)
    # initialize grid values. We have a separate grid for each velocity value
    grid_vR = np.empty(grid.shape)
    grid_vT = np.empty(grid.shape)
    grid_vz = np.empty(grid.shape)
    # get the grid value using sample V
    for i in range(R_number):
        for j in range(z_number):
            R, z = grid[i][j]
            vR, vT, vz = df.sampleV(R, z)
            grid_vR[i][j] = vR
            grid_vT[i][j] = vT
            grid_vz[i][j] = vz
    # generate interpolation objects
    ip_vR = interpolation(R_linspace, z_linspace, grid_vR)
    ip_vT = interpolation(R_linspace, z_linspace, grid_vT)
    ip_vz = interpolation(R_linspace, z_linspace, grid_vz)
    #break down normal into its R and z components
    normal_R = normal[:,0]
    normal_z = normal[:,1]
    # sample the velocity of normal points using interpolation of the grid
    normal_vR = ip_vR.ev(normal_R, normal_z)
    normal_vT = ip_vT.ev(normal_R, normal_z)
    normal_vz = ip_vz.ev(normal_R, normal_z)
    # putting together position coordinate with velocity coordinate for normal
    # points
    normal_coord_v = np.concatenate((normal_R, normal_z, normal_vR, 
                                     normal_vT, normal_vz))
    
    # combine normal and outlier result
    coord_v = np.vstack((normal_coord_v, outlier_coord_v))
    return coord_v
    
    
    
    
    
    
    
    
    
    