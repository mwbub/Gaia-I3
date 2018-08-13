import numpy as np
import sys
sys.path.append('../..')
sys.path.append('../../check_uniformity_of_density')

from main.main_program_cluster import main
qdf_sample = np.load("data/qdf sample cartesian physical, date=(2018, 8, 10).npy")

main(uniformity_method = "projection", gradient_method = "analytic",
     search_method = "local", custom_density = None, custom_samples = qdf_sample,
     custom_centres = None)