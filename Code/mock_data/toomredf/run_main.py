import sys
sys.path.append('../..')
sys.path.append('../../check_uniformity_of_density')

import numpy as np
from main.main_program_cluster import main
from sample_toomredf import sample_like_gaia
from toomredf import ToomrePotential

n = 4
epsilon = 1

data = sample_like_gaia(n, epsilon)
pot = ToomrePotential(n)
main(uniformity_method='projection', gradient_method='numeric', 
     custom_samples=data, custom_potential=pot)