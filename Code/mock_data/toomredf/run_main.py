import sys
sys.path.append('../..')
sys.path.append('../../main')
sys.path.append('../../check_uniformity_of_density')

import dill
import numpy as np
from main.main_program_cluster import main
from sample_toomredf import sample_like_gaia, sample_like_gaia_selection
from toomredf import ToomrePotential

file = '../../selection/parallax selection with galactic plane/selection_function'
with open(file, 'rb') as dill_file:
    selection = dill.load(dill_file)
    
file = r'D:\mwbub\Documents\AST299Y\Gaia-I3\Code\mock_data\toomredf\main_program_results\toomre_selection_epsilon=1\projection\projection data.npz'
with np.load(file) as data:
    cluster = data['cluster']

n = 4
epsilon = 1

data = sample_like_gaia_selection(n, epsilon)
pot = ToomrePotential(n)
main(uniformity_method='projection', gradient_method='numeric', 
     custom_samples=data, custom_potential=pot, custom_centres=cluster,
     selection=selection, band_width=5)