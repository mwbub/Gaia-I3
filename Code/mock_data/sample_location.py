"""
NAME:
    sample location

PURPOSE:
    A function that samples location under a given density function
    
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
        df = a density function that takes input in galactocentric Cartesian
             coordinate and return the normalized density at that point
             
        n = number of samples desired
        
        R_max, R_min = maximum and minimum radius (in natural units)
        
        z_max, z_min = maximum and minimum height (in natural units)
        
        phi_max, phi_min = maximum and minimum angle (in radian)

    OUTPUT:
        A numpy array in the form [(R, z, phi), (R, z, phi), ...] with n
        components.


    HISTORY:
        2018-07-08 - Written - Samuel Wong
    """
    # generate a random point in the cube [R_min, R_max]x[z_min, z_max]x[0,1]
    # this is effectively a random point in Rz space and a random trial
    # probability
    low =  (R_min, z_min, 0)
    high = (R_max, z_max, 1)
    R, z, p_trial = np.random.uniform(low, high)
    # calculate the actual probability at this point
    
    
    
    
    