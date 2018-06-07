#Ayush Pandhi, last updated 06/04/2018

#Importing the required modules
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.stats import iqr

#Defining a KDE function to quickly compute probabilities for the data set
def generate_KDE(inputs, ker, v_scale = 0.1):
    """
    Takes an NxM matrix for inputs and a ker string to output a function input_DKE that treats kde 
    as a black box function for sampling.
    Args:
        inputs (ndarray): NxM matrix, N = # of data points, M = # of parameters.
        ker (string): One of the 6 avaliable kernel types (gaussian, tophat, epanechnikov, exponential, linear, cosine)
        v_scale: velocity scaling argument
    Returns:
        kde (function):
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
        Takes a QxM matrix for samples to output a 1xQ array of density values.
        Args:
            samples (ndarray): QxM matrix, Q = # of points being evaluated, M = # of parameters.
        Returns:
            dens (ndarray): 1xQ array of density values for Q data points.
        """
        #To correct the type of information from other functions into acceptable input
        samples = [samples]
        
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

