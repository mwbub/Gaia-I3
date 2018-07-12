"""
NAME:
    kmeans

PURPOSE:
    Allows sampling by KMeans or Minibatch KMeans clustering.
    
HISTORY:
    2018-06-20 - Written - Michael Poon

"""

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from astropy.stats import median_absolute_deviation


def kmeans(samples, n_clusters, batch_size): 
    """
    NAME:
        kmeans
        
    PURPOSE:
        Allows sampling by KMeans or Minibatch KMeans clustering.
        
    INPUT:
        samples - Nx6 array of rectangular phase space coordinates of the form 
                  (x, y, z, vx, vy, vz) in [kpc, kpc, kpc, km/s, km/s, km/s]
        
        n_clusters - number of centroids generated from MiniBatch KMeans
        
        batch_size - batch size per iteration of MiniBatch KMeans over
                     gradient descent
        
        
    OUTPUT:
        Nx6 array of rectangular phase space coordinates of the form 
        (x, y, z, vx, vy, vz) in [kpc, kpc, kpc, km/s, km/s, km/s]
    """
    
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size)
    samples_mad = median_absolute_deviation(samples, axis=0, ignore_nan=True)
    kmeans.fit(samples/samples_mad)
    return kmeans.cluster_centers_*samples_mad

    
    