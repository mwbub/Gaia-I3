import numpy as np
from main_program_cluster import get_samples_density_filename, \ 
    get_Energy_Lz_gradient, evaluate_uniformity

def bootstrap(file, nsamples=10, uniformity_method='projection', 
              gradient_method='analytic', search_method='local', 
              custom_samples=None):
    """
    NAME:
        bootstrap
        
    PURPOSE:
        Run a bootstrap uncertainty analysis on the results of a run of the main
        program.
        
    INPUT:
        file - the results file of a run of the main program.
        
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
                            
    OUTPUT:
        None (results are saved to a file)
    """
    data = np.load(file)
    cluster_centres = data['cluster']
    samples, density, file_name = get_samples_density_filename(
            None, search_method, custom_samples, uniformity_method)
    
    for i in range(nsamples):
        

        