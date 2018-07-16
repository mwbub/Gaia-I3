import sys
sys.path.append('../..')
sys.path.append('../../check_uniformity_of_density')

from main.main_program_cluster_toomre import main
from sample_toomredf import sample_like_gaia

data = sample_like_gaia(4,4)
main(custom_samples=data, gradient_method='numeric')