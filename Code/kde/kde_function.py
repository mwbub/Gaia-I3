#Importing the required modules
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.stats import iqr

#Defining a KDE function to quickly compute probabilities for the data set
def generate_KDE(inputs, ker, v_scale):
    """
    NAME:
        generate_KDE
    
    PURPOSE:
        Given an NxM matrix for inputs, one of six avaliable ker strings 
        and a float value for v_scale to output a function `input_DKE` 
        that treats the density estimate as a black box function that 
        can be sampled.
    
    INPUT:
        inputs (ndarray) = An NxM matrix where N is the number of data 
                           points and M is the number of parameters.
        ker (string) = One of the 6 avaliable kernel types (gaussian, 
                       tophat, epanechnikov, exponential, linear, cosine).
        v_scale (float) = A float value to scale velocities for the kde.
    
    OUTPUT:
        input_KDE (function) = A blackbox function for the density estimate
                               used for sampling data.
                               
    HISTORY:
        2018-06-14 - Updated - Ayush Pandhi
    """
    #Scaling velocities with v_scale
    positions, velocities = np.hsplit(inputs, 2)
    velocities_scaled = velocities*v_scale
    inputs = np.hstack((positions, velocities_scaled))
    
    #Optimizing bandwidth in terms of Scott's Rule of Thumb
    shape_string = str(inputs.shape)
    objects, parameters = shape_string.split(', ')
    N_string = objects[1:]
    N = int(N_string)
    IQR = iqr(inputs)
    A = min(np.std(inputs), IQR/1.34)
    bw = 1.059 * A * N ** (-1/5.)
    
    #Fit data points to selected kernel and bandwidth
    kde = KernelDensity(kernel=ker, bandwidth=bw).fit(inputs)  

    def input_KDE(samples):
        """
        NAME:
            input_KDE
    
        PURPOSE:
            Given a QxM matrix for samples, evaluates the blackbox density
            estimate function at those points to output a 1xQ array of 
            density values.
    
        INPUT:
            samples (ndarray) = A QxM matrix where Q is the number of points 
                                at which the kde is being evaluated and M is 
                                the number of parameters.
                                
        OUTPUT:
            dens (ndarray) = A 1xQ array of density values for Q data points.
                               
        HISTORY:
            2018-06-14 - Updated - Ayush Pandhi
        """
        #To correct the type of information from other functions into acceptable input
        samples = np.array([samples])
        
        #Scaling samples with v_scale
        samp_positions, samp_velocities = np.hsplit(samples, 2)
        samp_velocities_scaled = samp_velocities*v_scale
        samples = np.hstack((samp_positions, samp_velocities_scaled))
        
        #Get the log density for selected samples and apply exponential to get normal probabilities
        log_dens = kde.score_samples(samples)
        dens = np.exp(log_dens)
        
        #Return a 1xQ array of normal probabilities for the selected sample
        return dens
    
    #Return a black box function for sampling
    return input_KDE
