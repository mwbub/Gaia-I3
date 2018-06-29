import sys
sys.path.append('../..')

import numpy as np
from main.main_program_cluster_psp import main
from sample_near_gaia import load_mock_data

data = load_mock_data(0, 0, 0, 3, parallax_cut=False)
data = data[~np.any(np.isnan(data), axis=1)]

main(None, 'local', data)
