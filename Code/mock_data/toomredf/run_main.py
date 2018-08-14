import sys
sys.path.append('../..')
sys.path.append('../../main')
sys.path.append('../../check_uniformity_of_density')

from main.main_program_cluster import main
from sample_toomredf import sample_like_gaia, sample_like_gaia_selection
from toomredf import ToomrePotential

n = 4
epsilon = 1

data = sample_like_gaia(n, epsilon)

x, y, z = data.T[:3]
mask = (x+8.3)**2 + y**2 + z**2 < 1**2
data = data[mask]

pot = ToomrePotential(n)
main(uniformity_method='projection', gradient_method='numeric', 
     custom_samples=data, custom_potential=pot)