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
    samples_std = np.nanstd(samples, axis=0)
    kmeans.fit(samples/samples_std)
    return kmeans.cluster_centers_*samples_std

    
    