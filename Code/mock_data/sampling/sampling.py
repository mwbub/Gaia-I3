"""
NAME:
    sampling

PURPOSE:
    A function that samples location under a given density function
    A function that samples velocity under a given velocity distribution
    
HISTORY:
    2018-07-08 - Written - Samuel Wong
"""
import numpy as np

def sample_location(df, n, R_min, R_max, z_min, z_max, phi_min, phi_max):
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
    #[R_min, R_max]x[z_min, z_max]x[0,1]
    # this is effectively random points in Rz space and a random trial
    # probability
    low =  (R_min, z_min, 0)
    high = (R_max, z_max, 1)
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
    
def sample_velocity(df, v_max, n):
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

    OUTPUT:
        a numpy array containing a list of n velocities 

    HISTORY:
        2018-07-13 - Written - Samuel Wong
    """
    # initialize list storing velocity result
    v_set = []
    # generate an array of random point in the cube: [0, v_max]x[0,1]
    low =  (0, 0)
    high = (v_max, 1)
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
    
    