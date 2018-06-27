"""
Filename: sample.py
Author: Mathew Bub
Last Revision Date: 2018-06-21

This module contains functions used to generate mock data using Dehnen DF.
The functions initially generate random sample stars in 2D, before adding a
z component to each star. This somewhat generalizes Dehnen DF to 3D. 
"""
import os
import time
import numpy as np
from sys import stdout
from galpy.df import dehnendf
from galpy.orbit import Orbit
from galpy.potential import MWPotential2014, LogarithmicHaloPotential
from galpy.util.bovy_coords import cyl_to_rect, cyl_to_rect_vec
from galpy.util.bovy_conversion import time_in_Gyr

_ERASESTR = '\r                                                              \r'

def get_samples_with_z(n=1, r_range=None, integration_time=1, 
                       integration_steps=100, use_psp=True):
    """
    NAME:
        get_samples_with_z
    
    PURPOSE:
        sample stars with R, vR, and vT using Dehnen DF, then add a z and vz
        component to each sampled star
        
    INPUT:
        n - number of samples to generate (optional; default = 1)
        
        r_range - radial range in kpc in which to sample stars; if None, will 
        sample stars at any radius (optional; default = None)
        
        integration_time - length of time to integrate orbits in Gyr; used for
        adding a z component to each star (optional; default = 1)
        
        integration_steps - number of steps to use in the orbit integration
        (optional; default = 100)
        
        use_psp - if True, will use LogarithmicHaloPotential for orbit 
        integration instead of MWPotential2014 (optional; default = True)
        
    OUTPUT:
        nx5 array of cylindrical galactocentric coordinates of the form
        (R, vR, vT, z, vz) in [kpc, km/s, km/s, kpc, km/s]
    """
    import warnings
    warnings.filterwarnings('ignore')
    
    stdout.write('sampling orbits...')
    if n >= 1000:
        average_time = get_average_sample_time(r_range=r_range)
        time_str = estimate_completion_time(n, average_time)
        stdout.write('\nestimated time of completion: ' + time_str)
        
    # convert r_range to natural units
    if r_range is not None:
        r_range = [r/8. for r in r_range]
        
    # sample R, vR, and vT over r_range
    df = dehnendf()
    sampled_ROrbits = df.sample(n=n, rrange=r_range)
    
    stdout.write('\ndone at ' + time.strftime('%H:%M:%S', time.localtime()))
    
    # get the R, vR, and vT values from each sampled orbit
    R = np.array([o.R() for o in sampled_ROrbits])
    vRz = np.array([o.vR() for o in sampled_ROrbits])
    vT = np.array([o.vT() for o in sampled_ROrbits])
    
    # divide radial kinetic energy evenly between the R and z directions
    kRz = vRz**2/2
    kR = kRz * np.random.random(n)
    kz = kRz - kR
    
    # convert kinetic energy to velocities, choosing a random starting 
    # direction for vz and retaining the original direction of vR
    vR = np.sqrt(2*kR) * np.sign(vRz)
    vz = np.sqrt(2*kz) * np.random.choice((1, -1), size=n)
    
    # store the coordinates in a format suitable for orbit objects
    coord = np.stack([[R[i], vR[i], vT[i], 0, vz[i]] for i in range(n)], axis=0)
    
    # delete the original orbits to preserve memory
    del sampled_ROrbits
    
    # integrate the orbits for integration_time Gyr
    t = np.linspace(0, integration_time/time_in_Gyr(vo=220., ro=8.), 
                    integration_steps)
    
    if use_psp:
        pot = LogarithmicHaloPotential(normalize=1.)
    else:
        pot = MWPotential2014
    
    stdout.write('\n\nintegrating orbits...\n')
    
    if n >= 1000:
        start = time.time()
        
    end = t[-1]
    for i in range(n):
        o = Orbit(vxvv=coord[i], ro=8., vo=220.)
        o.integrate(t, pot)
        coord[i] = [o.R(end), o.vR(end), o.vT(end), o.z(end), o.vz(end)]
        
        if n >= 1000 and i % 1000 == 0:
            average_time = (time.time() - start)/(i + 1)
            time_str = estimate_completion_time(n-i, average_time)
            stdout.write(_ERASESTR)
            stdout.write('estimated time of completion: ' + time_str)
    
    if n >= 1000:
        stdout.write('\n')
    stdout.write('done at ' + time.strftime('%H:%M:%S', time.localtime()))
    
    return coord

def generate_sample_data(n, phi_range, r_range=None, use_psp=True):
    """
    NAME:
        generate_sample_data
        
    PURPOSE:
        generate a sample of stars using Dehnen DF
        
    INPUT:
        n - number of samples to generate
        
        phi_range - phi range in radians over which to distribute the samples
        
        r_range - radial range in kpc in which to sample stars; if None, will 
        sample stars at any radius (optional; default = None)
        
        use_psp - if True, will use LogarithmicHaloPotential for orbit 
        integration instead of MWPotential2014 (optional; default = True)
        
    OUTPUT:
        None (saves samples to the data directory)
    """
    # sample orbits over r_range and phi_range
    samples = get_samples_with_z(n=n, r_range=r_range, use_psp=use_psp)
    phi = np.random.uniform(*phi_range, n)
    samples = np.concatenate((samples, phi.reshape((-1, 1))), axis=1)
    
    # convert to rectangular
    rect = np.stack(cyl_to_rect(*samples[:, [0,5,3]].T), axis=1)
    rect_vec = np.stack(cyl_to_rect_vec(*samples[:, [1,2,4,5]].T), axis=1)
    samples = np.concatenate((rect, rect_vec), axis=1)
    
    # create a directory to hold the samples
    if not os.path.exists('data'):
        os.mkdir('data')
    
    # choose a file name representing the chosen parameters
    if r_range is not None:
        filename = ('{}samples_{}-{}rad_{}-{}kpc'
                    ).format(n, *phi_range, *r_range)
    else:
        filename = '{}samples_{}-{}rad'.format(n, *phi_range)
        
    if use_psp:
        filename += '_psp'
        
    np.save('data/' + filename, samples)
    
def load_samples(n, phi_range, r_range=None, use_psp=True):
    """
    NAME:
        load_samples
        
    PURPOSE:
        load a sample of stars; if the sample does not exist in the data
        directory, generate it first
        
    INPUT:
        n - number of samples to generate
        
        phi_range - phi range in degrees over which to distribute the samples
        
        r_range - radial range in kpc in which to sample stars; if None, will 
        sample stars at any radius (optional; default = None)
        
        use_psp - if True, will use LogarithmicHaloPotential for orbit 
        integration instead of MWPotential2014 (optional; default = True)
        
    OUTPUT:
        nx6 array of rectangular galactocentric coordinates of the form 
        (x, y, z, vx, vy, vz) in [kpc, kpc, kpc, km/s, km/s, km/s],
        representing sampled stars
    """
    # choose a file name representing the chosen parameters
    if r_range is not None:
        filename = ('{}samples_{}-{}rad_{}-{}kpc'
                    ).format(n, *phi_range, *r_range)
    else:
        filename = '{}samples_{}-{}rad'.format(n, *phi_range)
        
    if use_psp:
        filename += '_psp'
        
    # check if the file already exists
    if not os.path.exists('data/' + filename + '.npy'):
        # generate the file if it does not exist
        generate_sample_data(n=n, phi_range=phi_range, r_range=r_range, 
                             use_psp=use_psp)
        
    # load the samples
    samples = np.load('data/' + filename + '.npy')
    return samples

def get_average_sample_time(r_range=None):
    """
    NAME:
        get_average_sample_time
        
    PURPOSE:
        estimate the average time to sample orbits in r_range
        
    INPUT:
        r_range - radial range in kpc in which to sample stars; if None, will 
        sample stars at any radius (optional; default = None)
        
    OUTPUT:
        average sample time
    """
    import warnings
    warnings.filterwarnings('ignore')
    
    if r_range is not None:
        r_range = [r/8. for r in r_range]
    
    df = dehnendf()
    
    # get time to sample 100 stars
    start = time.time()
    df.sample(n=1000, rrange=r_range)
    end = time.time()
    
    return (end-start)/1000

def estimate_completion_time(n, average_time):
    """
    NAME:
        estimate_completion_time
        
    PURPOSE:
        return the estimated completion time of n iterations of a process that
        takes average_time per iteration
        
    INPUT:
        n - number of iterations
        
        average_time - average time to complete each iteration
        
    OUTPUT:
        string representing estimated time of completion; if the completion
        time is more than 24 hours in the future, also output the date of
        completion
    """
    time_format = '%H:%M:%S'
    
    # add date to output if completion time is more than 24 hours away
    if average_time * n > 86400:
        time_format = '%d %b ' + time_format
        
    # estimate completion time
    completion_time = time.time() + average_time * n
    
    time_str = time.strftime(time_format, time.localtime(completion_time))
    return time_str