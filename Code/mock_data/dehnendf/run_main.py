import sys
sys.path.append('../..')

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from main.main_program_cluster_psp import main
from sample_near_gaia import load_mock_data

data = load_mock_data(0, 0, 0, 0.5, use_psp=True)
data = data[~np.any(np.isnan(data), axis=1)]
'''
galcen = SkyCoord(frame='galactocentric', representation_type='cartesian',
                  x=data.T[0], y=data.T[1], z=data.T[2], 
                  unit=[u.kpc, u.kpc, u.kpc])
gal = galcen.transform_to('galactic')
gal.representation_type = 'cartesian'
mask = (gal.u.value**2 + gal.v.value**2 + gal.w.value**2 < 0.5**2)
data = data[mask]
'''

main(None, 'local', data)
