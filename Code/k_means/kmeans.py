"""
NAME:
    kmeans

PURPOSE:
    Allows sampling by KMeans or Minibatch KMeans clustering.
    
HISTORY:
    2018-06-20 - Written - Michael Poon

"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans


def kmeans(samples, minibatch=True, scale='std'): 
    """
    NAME:
        kmeans
        
    PURPOSE:
        Allows sampling by KMeans or Minibatch KMeans clustering.
        
    INPUT:
        minibatch - if True, Minibatch KMeans clustering will be used.
        Faster runtime, but less accurate clustering (optional; default = True)
        
        samples - Nx6 array of rectangular phase space coordinates of the form 
        (x, y, z, vx, vy, vz) in [kpc, kpc, kpc, km/s, km/s, km/s]
        
        scale - To make 6D samples "spherical" or within a similar length or range,
                divide each coordinate by its standard deviation ('std') or 
                interquartile range ('iqr'). 
                
                This is done before and after clustering.
                
                (default = 'std')
        
    OUTPUT:
        Nx6 array of rectangular phase space coordinates of the form 
        (x, y, z, vx, vy, vz) in [kpc, kpc, kpc, km/s, km/s, km/s]
    """