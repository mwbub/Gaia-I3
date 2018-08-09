import os
import sys
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import KFold
from main_program_cluster import get_samples_density_filename, \
    get_Energy_Lz_gradient, evaluate_uniformity, generate_KDE, error_plot

_ERASESTR = '\r                                                              \r'


def bootstrap(nsamples=10, uniformity_method='projection', 
              gradient_method='analytic', search_method='local', 
              custom_samples=None, custom_potential=None):
    """
    NAME:
        bootstrap
        
    PURPOSE:
        Run a bootstrap uncertainty analysis on the results of a run of the main
        program.
        
    INPUT:
        nsamples - number of random samples to generate on which to re-run the
        main program; default = 10
        
        uniformity_method - 'projection' or 'dot product', referring to how
        uniformity of density is evaluated; default = projection
        
        gradient_method - 'analytic' or 'numeric', referring to how gradient of 
        energy and L_z are generated; default = 'analytic'
        
        search_method - search the gaia catalogue online ('online'), locally on 
        a downloaded file ('local'), or use the entire downloaded gaia rv file 
        ('all of local')
        
        custom_samples - an N by 6 array that represents the custom samples, 
        with each component representing (x,y,z,vx,vy,vz), respectively. They 
        are in physical units.
        
        custom_potential - galpy Potential or list of Potentials used to 
        evaluate energy; default = MWPotential2014
                            
    OUTPUT:
        None (results are saved to a file)
        
    WARNING:
        In order to use this function, you must first run the main program with
        the same parameters and in the same location as will be used to run this
        function.
    """
    samples, density, folder = get_samples_density_filename(
            None, search_method, custom_samples, uniformity_method)
    
    data_file = 'data.npz'
    if uniformity_method == 'projection':
        data_file = 'projection ' + data_file
        
    with np.load('main_program_results/' + folder + data_file) as data:
        cluster_centres = data['cluster']
        original_result = data['result']
    
    Energy_gradient, Lz_gradient = get_Energy_Lz_gradient(
            cluster_centres, gradient_method, custom_potential)
    
    results = []
    sys.stdout.write('\n')
    for i in range(nsamples):
        sys.stdout.write(_ERASESTR)
        sys.stdout.write('Evaluating uniformity on sample {}...'.format(i+1))
        resampled_data = resample(samples)
        density = generate_KDE(resampled_data, 'epanechnikov')
        result = evaluate_uniformity(density, cluster_centres, Energy_gradient,
                                     Lz_gradient, uniformity_method)
        results.append(result)
    sys.stdout.write('\nDone\n')
    
    results = np.stack(results)
    errors = np.nanstd(results, axis=0)
    
    folder += 'uncertainties/'
    if not os.path.exists('main_program_results/' + folder):
        os.mkdir('main_program_results/' + folder)
        
    folder += 'bootstrap/'
    if not os.path.exists('main_program_results/' + folder):
        os.mkdir('main_program_results/' + folder)
        
    folder += '{} samples/'.format(nsamples)
    if not os.path.exists('main_program_results/' + folder):
        os.mkdir('main_program_results/' + folder)
    
    np.save('main_program_results/' + folder + 'uncertainties', errors)
    
    # the max dot products of the original run of the main program are not
    # necessarily at the same index in the resampled runs, and thus their
    # uncertainties must be calculated separately from the uncertainties in each
    # individual dot product
    if uniformity_method == 'dot product':
        results = np.nanmax(np.abs(results), axis=2)
        errors = np.nanstd(results, axis=0)
        np.save('main_program_results/' + folder + 
                'max_dot_product_uncertainties', errors)
        
    error_plot(errors, cluster_centres, folder, uniformity_method, 
               custom_potential)
    errorbar_plot(original_result, cluster_centres, folder, uniformity_method,
                  custom_potential, errors)
    
    
def jackknife(nsamples=10, uniformity_method='projection', 
              gradient_method='analytic', search_method='local', 
              custom_samples=None, custom_potential=None):
    """
    NAME:
        jackknife
        
    PURPOSE:
        Run a jackknife uncertainty analysis on the results of a run of the main
        program.
        
    INPUT:
        nsamples - number of groups in which to divide the data and on which to 
        re-run the main program; default = 10
        
        uniformity_method - 'projection' or 'dot product', referring to how
        uniformity of density is evaluated; default = projection
        
        gradient_method - 'analytic' or 'numeric', referring to how gradient of 
        energy and L_z are generated; default = 'analytic'
        
        search_method - search the gaia catalogue online ('online'), locally on 
        a downloaded file ('local'), or use the entire downloaded gaia rv file 
        ('all of local')
        
        custom_samples - an N by 6 array that represents the custom samples, 
        with each component representing (x,y,z,vx,vy,vz), respectively. They 
        are in physical units.
        
        custom_potential - galpy Potential or list of Potentials used to 
        evaluate energy; default = MWPotential2014
                            
    OUTPUT:
        None (results are saved to a file)
        
    WARNING:
        In order to use this function, you must first run the main program with
        the same parameters and in the same location as will be used to run this
        function.
    """
    samples, density, folder = get_samples_density_filename(
            None, search_method, custom_samples, uniformity_method)
    
    data_file = 'data.npz'
    if uniformity_method == 'projection':
        data_file = 'projection ' + data_file
        
    with np.load('main_program_results/' + folder + data_file) as data:
        cluster_centres = data['cluster']
        original_result = data['result']
    
    Energy_gradient, Lz_gradient = get_Energy_Lz_gradient(
            cluster_centres, gradient_method, custom_potential)
    
    # split the data into nsamples non-overlapping splits, each of which contain
    # (len(samples) - len(samples) / nsamples) points
    kf = KFold(n_splits=nsamples, shuffle=True)
    splits = kf.split(samples)
    split_indices = [indices for indices, _ in splits]
    
    results = []
    sys.stdout.write('\n')
    for i in range(len(split_indices)):
        sys.stdout.write(_ERASESTR)
        sys.stdout.write('Evaluating uniformity on sample {}...'.format(i+1))
        split = split_indices[i]
        density = generate_KDE(samples[split], 'epanechnikov')
        result = evaluate_uniformity(density, cluster_centres, Energy_gradient,
                                     Lz_gradient, uniformity_method)
        results.append(result)
    sys.stdout.write('\nDone\n')
    
    results = np.stack(results)
    var = np.sum((results-original_result)**2, axis=0)*(nsamples-1)/nsamples
    errors = np.sqrt(var)
    
    folder += 'uncertainties/'
    if not os.path.exists('main_program_results/' + folder):
        os.mkdir('main_program_results/' + folder)
        
    folder += 'jackknife/'
    if not os.path.exists('main_program_results/' + folder):
        os.mkdir('main_program_results/' + folder)
        
    folder += '{} samples/'.format(nsamples)
    if not os.path.exists('main_program_results/' + folder):
        os.mkdir('main_program_results/' + folder)
    
    np.save('main_program_results/' + folder + 'uncertainties', errors)
    
    # the max dot products of the original run of the main program are not
    # necessarily at the same index in the resampled runs, and thus their
    # uncertainties must be calculated separately from the uncertainties in each
    # individual dot product
    if uniformity_method == 'dot product':
        results = np.nanmax(np.abs(results), axis=2)
        original_result = np.nanmax(np.abs(original_result), axis=1)
        var = np.sum((results-original_result)**2, axis=0)*(nsamples-1)/nsamples
        errors = np.sqrt(var)
        np.save('main_program_results/' + folder + 
                'max_dot_product_uncertainties', errors)
        
    error_plot(errors, cluster_centres, folder, uniformity_method, 
               custom_potential)
    errorbar_plot(original_result, cluster_centres, folder, uniformity_method,
                  custom_potential, errors)