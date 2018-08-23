import numpy as np
import sys
import dill
sys.path.append('../..')
sys.path.append('../../check_uniformity_of_density')

from main.main_program_cluster import main
qdf_sample = np.load("data/qdf sample cartesian physical with new selection, e=1, date=(2018, 8, 20).npy")
qdf_sample[:,0] = -qdf_sample[:,0] # fix orientation by adding negative sign in front of x
with open("../../selection/parallax selection with galactic plane/selection_function","rb") as dill_file:
    selection = dill.load(dill_file)

main(uniformity_method = "projection", gradient_method = "analytic",
         search_method = "local", custom_density = None, custom_samples = qdf_sample,
         custom_centres = None, custom_potential = None,
         selection = selection, band_width = 5)