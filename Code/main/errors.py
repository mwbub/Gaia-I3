import numpy as np
from main_program_cluster import get_samples_density_filename, \
    get_Energy_Lz_gradient, evaluate_uniformity, generate_KDE

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
        
        custom_potential = galpy Potential or list of Potentials; default = 
        MWPotential2014
                            
    OUTPUT:
        None (results are saved to a file)
        
    WARNING:
        In order to use this function, you must first run the main program with
        the same parameters and in the same location as will be used to run this
        function.
    """
    samples, density, path = get_samples_density_filename(
            None, search_method, custom_samples, uniformity_method)
    path = 'main_program_results/' + path
    
    file = 'data.npz'
    if uniformity_method == 'projection':
        file = 'projection ' + file
        
    data = np.load(path + file)
    cluster_centres = data['cluster']
    
    Energy_gradient, Lz_gradient = get_Energy_Lz_gradient(
            cluster_centres, gradient_method, custom_potential)
    
    results = []
    for i in range(nsamples):
        resampled_indices = np.random.choice(samples.shape[0], samples.shape[0])
        density = generate_KDE(samples[resampled_indices], 'epanechnikov')
        result = evaluate_uniformity(density, cluster_centres, Energy_gradient,
                                     Lz_gradient, uniformity_method)
        results.append(result)

    results = np.stack(results)
    return results