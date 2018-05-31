
# coding: utf-8

# In[38]:


#Ayush Pandhi, last updated 05/30/2018

#Importing the required modules
import numpy as np
import random
from scipy.stats import iqr

#Defining a function to optimize bandwidth based on Scott's rule of thumb
def Scott_bw(N, X):
    """
    Takes N (number of objects) and X (NxM matrix) to output the optimal bandwidth based on Scott's rule of thumb.
    
    Args:
        N (float): Number of objects.
        X (ndarray): NxM matrix, N = # of objects, M = # of parameters.
    Returns:
        bw (float): The optimized bandwidth as a float number.
    
    """
    IQR = iqr(X)
    A = min(np.std(X), IQR/1.34)
    bw = 1.059 * A * N ** (-1/5.)
    return bw

