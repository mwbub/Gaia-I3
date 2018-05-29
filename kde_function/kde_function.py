#Ayush Pandhi, last updated 05/28/2018

# Importing the required modules
import numpy as np
from sklearn.neighbors import KernelDensity


# Defining a KDE function to quickly compute probabilities for the data set
def generate_KDE(inputs, ker, bw):
    """
    Takes an NxM matrix for inputs, a string for ker and a float for bw to output
    a function input_DKE that treats kde as a black box function for sampling.

    Args:
        inputs (ndarray): NxM matrix, N = # of data points, M = # of parameters.
        ker (string): One of the 6 available kernel types (gaussian, tophat, epanechnikov, exponential, linear, cosine)
        bw (float): Bandwidth of the kernel as a dimensionless float.
    Returns:
        kde (function):

    """
    kde = KernelDensity(kernel=ker, bandwidth=bw).fit(inputs)  # Fit data points to selected kernel and bandwidth

    def input_KDE(samples):
        """
        Takes an NxM matrix for inputs and a QxM matrix for samples, a string for ker and a float for bw to output
        a 1xQ array of density values.

        Args:
            samples (ndarray): QxM matrix, Q = # of points being evaluated, M = # of parameters.
        Returns:
            dens (ndarray): 1xQ array of density values for Q data points.

        """
        log_dens = kde.score_samples(samples)  # Get the log density for the selected samples
        dens = np.exp(log_dens)  # Apply exponential to get normal density from log
        return dens  # Return a 1xQ array of probabilities
    return input_KDE
