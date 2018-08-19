#Importing the required modules
import numpy as np
from sklearn.neighbors import KernelDensity
from astropy.coordinates import SkyCoord

#Defining a KDE function to quickly compute probabilities for the data set
def generate_KDE(inputs, ker, selection = None, bw_multiplier=10):
    """
    NAME:
        generate_KDE
    
    PURPOSE:
        Given an NxM matrix for inputs and one of six avaliable ker strings, 
        outputs a function `input_DKE` that treats the density estimate as a 
        black box function that can be sampled.
    
    INPUT:
        inputs (ndarray) = An NxM matrix where N is the number of data 
                           points and M is the number of parameters.
        ker (string) = One of the 6 avaliable kernel types (gaussian, 
                       tophat, epanechnikov, exponential, linear, cosine).
        selection = a selection function that takes parallax to Sun and returns
                    fraction of stars that are left after selection;
                    takes array; takes parallax in physical units
    
    OUTPUT:
        input_KDE (function) = A blackbox function for the density estimate
                               used for sampling data.
                               
    HISTORY:
        2018-07-15 - Updated - Ayush Pandhi
    """
    #Scaling velocities with z-score
    inputs_std = np.nanstd(inputs, axis=0)
    i1, i2, i3, i4, i5, i6 = np.mean(inputs, axis=0)
    inputs_mean = np.hstack((i1, i2, i3, i4, i5, i6))
    inputs = (inputs - inputs_mean)/inputs_std
    
    #Optimizing bandwidth in terms of Scott's Multivariate Rule of Thumb
    N = inputs.shape[0]
    bw = bw_multiplier * np.nanstd(inputs) * N ** (-1/10.)
    
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
            2018-07-15 - Updated - Ayush Pandhi
        """
        if selection is not None:
            #compute parallax in physical units. Inputs are in natural units.
            x, y, z, vx, vy, vz = samples.T
            #distannce to sun; compute sqrt in natural unit; times 8 to turn to
            #physical
            distance = 8.*np.sqrt((x-(-1.03749451))**2 + y**2 + (z-0.000875)**2)
            parallax = 1/distance
            # convert cartesian galactocentric to galactic
            gal = SkyCoord(x=8.*x, y=8.*y, z=8.*z, unit="kpc",
                                frame="galactocentric").galactic
            b = gal.b.degree
        
        #Scaling samples with standard deviation
        samples = (samples - inputs_mean)/inputs_std
        
        #Get the log density for selected samples and apply exponential to get normal probabilities
        log_dens = kde.score_samples(samples)
        dens = np.exp(log_dens)
        
        #Return a 1xQ array of normal probabilities for the selected sample
        if selection is None:
            return dens
        else:
            # divide by selection fraction only when selection function is given
            return dens/selection(parallax, b)
    
    #Return a black box function for sampling
    return input_KDE
