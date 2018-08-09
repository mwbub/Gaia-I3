"""
NAME:
    sampling

PURPOSE:
    A function that samples location under a given density function
    A function that samples location under a given density function and using
    interpolation for faster performance
    A function that samples velocity under a given velocity distribution
    
HISTORY:
    2018-07-08 - Written - Samuel Wong
    2018-07-14 - Added interpolation sampling - Samuel Wong
"""
import numpy as np
from scipy import interpolate

def sample_location(df, n, R_min, R_max, z_min, z_max, phi_min, phi_max, df_max):
    """
    NAME:
        sample_location

    PURPOSE:
        Given a density function and the maximum and minimum in cylindrical 
        coordinate, as well as number of samples desired, sample location.
        
    INPUT:
        df = a density function that takes input in galactocentric Cylindrical
             coordinate and return the normalized density at that point;
             Assume it takes array in the form 
             df(array([R,R,..]),array([z,z,...]))
             
        n = number of samples desired
        
        R_max, R_min = maximum and minimum radius (in natural units)
        
        z_max, z_min = maximum and minimum height (in natural units)
        
        phi_max, phi_min = maximum and minimum angle (in radian)
        
        df_max = maximum value of the dsitribution function

    OUTPUT:
        A numpy array in the form [(R, z, phi), (R, z, phi), ...] with n
        components.


    HISTORY:
        2018-07-08 - Written - Samuel Wong
        2018-07-13 - Modified so that df assumed to take arrays - Samuel Wong
    """
    # initialize list storing [[R,z], [R,z],..] result
    Rz_set = []
    # generate an array of random point in the cube:
    #[R_min, R_max]x[z_min, z_max]x[0,df_max]
    # this is effectively random points in Rz space and a random trial height
    low =  (R_min, z_min, 0)
    high = (R_max, z_max, df_max)
    # repeat while not enough points are generated yet
    while len(Rz_set) < n:
        # number of points to generate is the number of points missing
        nmore = n - len(Rz_set)
        # generate randome points in cube
        R_z_ptrial = np.random.uniform(low, high, size=(nmore, 3))
        R, z, p_trial = [R_z_ptrial[:, i] for i in range(3)]
        # calculate the actual probability at these points
        p = df(R,z)
        # accept point if trial is less than real probability; in other words,
        # accept if the point is below the curve
        mask = p_trial < p
        R_accept = R[mask]
        z_accept = z[mask]
        Rz_accept = np.stack((R_accept, z_accept), axis = 1)
        # add accepted points into stored list
        Rz_set += Rz_accept.tolist()
    # convert Rz set to array
    Rz_set = np.array(Rz_set)
    # get a unifrom distribution in phi
    phi_set = np.reshape(np.random.uniform(phi_min, phi_max, n), (n, 1))
    Rzphi_set = np.hstack((Rz_set, phi_set))
    return Rzphi_set


def sample_location_selection(df, n, R_min, R_max, z_min, z_max, phi_min,
                              phi_max, df_max, selection, R_0 = 8.3, z_0 = 0.,
                              phi_0 = 0.):
    """
    NAME:
        sample_location_selection

    PURPOSE:
        Given a density function and the maximum and minimum in cylindrical 
        coordinate, as well as number of samples desired, sample location.
        
    INPUT:
        df = a density function that takes input in galactocentric Cylindrical
             coordinate and return the normalized density at that point;
             Assume it takes array in the form 
             df(array([R,R,..]),array([z,z,...]))
             
        n = number of samples desired
        
        R_max, R_min = maximum and minimum radius (in natural units)
        
        z_max, z_min = maximum and minimum height (in natural units)
        
        phi_max, phi_min = maximum and minimum angle (in radian)
        
        df_max = maximum value of the dsitribution function
        
        selection = a selection function that takes parallax to Sun; takes array
        
        R_0, z_0 = Sun's location

    OUTPUT:
        A numpy array in the form [(R, z, phi), (R, z, phi), ...] with n
        components.


    HISTORY:
        2018-08-08 - Written - Samuel Wong
    """
    # initialize list storing [[R,z,phi], [R,z,phi],..] result
    Rzphi_set = []
    # generate an array of random point in the cube:
    #[R_min, R_max]x[z_min, z_max]x[phi_min, phi_max]x[0,df_max]x[0,1]
    # this is effectively random points in 5 dimensional space
    low =  (R_min, z_min, phi_min, 0, 0)
    high = (R_max, z_max, phi_max, df_max, 1)
    # repeat while not enough points are generated yet
    while len(Rzphi_set) < n:
        # number of points to generate is the number of points missing
        nmore = n - len(Rzphi_set)
        # generate randome points in cube
        R_z_phi_ptrial_psel = np.random.uniform(low, high, size=(nmore, 5))
        R, z, phi, p_trial, p_sel = [R_z_phi_ptrial_psel[:, i] for i in range(5)]
        # calculate the actual probability at these points
        p = df(R,z)
        # first mask: if trial is less than real probability; in other words,
        # accept if the point is below the curve
        mask1 = p_trial < p
        # second mask: selection function
        # calculate distance to Sun
        distance = np.sqrt(R**2 + R_0**2 - 2*R*R_0*np.cos(phi-phi_0) + (z - z_0)**2)
        parallax = 1/distance
        mask2 = p_sel < selection(parallax)
        #accept
        R_accept = R[np.all(np.array([mask1, mask2]), axis = 0)]
        z_accept = z[np.all(np.array([mask1, mask2]), axis = 0)]
        phi_accept = phi[np.all(np.array([mask1, mask2]), axis = 0)]
        Rzphi_accept = np.stack((R_accept, z_accept, phi_accept), axis = 1)
        # add accepted points into stored list
        Rzphi_set += Rzphi_accept.tolist()
    # convert Rzphi set to array
    Rzphi_set = np.array(Rzphi_set)
    return Rzphi_set


def sample_location_interpolate(df, n, R_min, R_max, z_min, z_max, phi_min,
                                phi_max, df_max, pixel_R, pixel_z):
    """
    NAME:
        sample_location_interpolate

    PURPOSE:
        Given a density function and the maximum and minimum in cylindrical 
        coordinate, as well as number of samples desired, sample location using
        interpolation.
        
    INPUT:
        df = a density function that takes input in galactocentric Cylindrical
             coordinate and return the normalized density at that point;
             Assume it takes array in the form 
             df(array([R,R,..]),array([z,z,...]))
             
        n = number of samples desired
        
        R_max, R_min = maximum and minimum radius (in natural units)
        
        z_max, z_min = maximum and minimum height (in natural units)
        
        phi_max, phi_min = maximum and minimum angle (in radian)
        
        df_max = maximum value of the dsitribution function
        
        pixel_R, pixel_z = the distance in R and z when making grid for
                           interpolation

    OUTPUT:
        A numpy array in the form [(R, z, phi), (R, z, phi), ...] with n
        components.


    HISTORY:
        2018-07-14 - Written - Samuel Wong
    """
    # calculate the number of spacing in each direction
    R_number = int((R_max - R_min)/pixel_R)
    z_number = int((z_max - z_min)/pixel_z)
    #create the linspace in each direction according to the specified number
    #of points in each axis
    R_linspace = np.linspace(R_min, R_max, R_number)
    z_linspace = np.linspace(z_min, z_max, z_number)
    # mesh and create the grid
    Rv, zv = np.meshgrid(R_linspace, z_linspace)
    #get grid
    grid = np.dstack((Rv, zv))
    #initialize grid values.
    #grid is a 3 dimensional array since it stores pairs of values, but 
    #grid values are 2 dimensinal array
    grid_df = np.empty((grid.shape[0], grid.shape[1]))
    #find df value on the grid
    for i in range(z_number):
        for j in range(R_number):
            R, z = grid[i][j]
            grid_df[i][j] = df(R,z)
    # generate interpolation object
    ip_df = interpolate.RectBivariateSpline(z_linspace, R_linspace,
                                                grid_df)
    # since the interpolation object takes (z,R) coordinate, define a 
    #wrapper around the interpoaltion evaluation function
    def evaluate_ip(R, z):
        return ip_df.ev(z, R)
    # call the normal sampling location function and pass the interpolation
    # evaluation as df
    return sample_location(evaluate_ip, n, R_min, R_max, z_min, z_max, phi_min,
                                phi_max, df_max)
    
def sample_velocity(df, v_max, n, df_max):
    """
    NAME:
        sample_location

    PURPOSE:
        Given a density function and the maximum and minimum in cylindrical 
        coordinate, as well as number of samples desired, sample location.
        
    INPUT:
        df = a distribution function that takes an array of velocity and
            output their probability density
             
        v_max = maximum velocity allowed
        
        n = number of samples desired
        
        df_max = maximum value of the dsitribution function

    OUTPUT:
        a numpy array containing a list of n velocities 

    HISTORY:
        2018-07-13 - Written - Samuel Wong
    """
    # initialize list storing velocity result
    v_set = []
    # generate an array of random point in the cube: [0, v_max]x[0,df_max]
    low =  (0, 0)
    high = (v_max, df_max)
    # repeat while not enough points are generated yet
    while len(v_set) < n:
        # number of points to generate is the number of points missing
        nmore = n - len(v_set)
        # generate randome points in cube
        v_ptrial = np.random.uniform(low, high, size=(nmore, 2))
        v, p_trial = [v_ptrial[:, i] for i in range(2)]
        # calculate the actual probability at these points
        p = df(v)
        # accept point if trial is less than real probability; in other words,
        # accept if the point is below the curve
        mask = p_trial < p
        v_accept = v[mask]
        v_set += v_accept.tolist()
    # convert v set to array
    v_set = np.array(v_set)
    return v_set
    
    