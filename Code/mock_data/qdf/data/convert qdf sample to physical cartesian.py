import numpy as np
import os, sys
sys.path.append('../../../tools')
from tools import cyl_to_rect, to_physical_units

coord_cyl = np.load("qdf sample data.npy")
R, z, phi, vR, vT, vz = coord_cyl.T
# convert to cartesian
coord_rect = cyl_to_rect(R, vR, vT, z, vz, phi)
# convert from natural to physics 
physical_rect = to_physical_units(coord_rect)
#save data
np.save("qdf sample cartesian physical.npy", physical_rect)