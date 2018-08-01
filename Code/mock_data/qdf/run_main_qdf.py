import numpy as np
import sys
sys.path.append('../..')
sys.path.append('../../check_uniformity_of_density')

from main.main_program_cluster import main
qdf_sample = np.load("data/qdf sample cartesian physical.npy")

main(uniformity_method = "projection", gradient_method = "analytic",
     search_method = "local", custom_samples = qdf_sample)