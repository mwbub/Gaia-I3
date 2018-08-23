import sys
sys.path.append('../sampling')
sys.path.append('../../tools')
from tools import cyl_to_rect, to_physical_units
from sampling import sample_location_interpolate_selection
import numpy as np
import dill
import time as time_class
#import qdf related things
from galpy.potential import MWPotential2014
from galpy.actionAngle import actionAngleAdiabatic
from galpy.df import quasiisothermaldf
aA= actionAngleAdiabatic(pot=MWPotential2014,c=True)
# set up qdf
# qdf not imported from galpy since I am running a modified galpy in console
qdf= quasiisothermaldf(1./3.,0.2,0.1,1.,1.,pot=MWPotential2014,aA=aA,cutcounter=True)
#get selection function
with open("../../selection/parallax selection with galactic plane/selection_function", "rb") as dill_file:
    selection_function = dill.load(dill_file)

start = time_class.time()
#sample location
# get the maximum of qdf
qdf_max = 1.2*qdf.density(0.875, 0)
# sample R from 7.5 kpc to 8.5 kpc; sample z from -0.5 kpc to 0.5 kpc
# let phi range from -arctan(0.5/8) to arctan(0.5/8)
location = sample_location_interpolate_selection(df=qdf.density,n=3100000,
                                                 R_min=0.875, R_max=1.125,
                                                 z_min=-0.125, z_max=0.125,
                                                 phi_min=-0.124,
                                                 phi_max=0.124,
                                                 df_max = qdf_max,
                                                 pixel_R = 0.01,
                                                 pixel_z = 0.01,
                                                 selection=selection_function,
                                                 R_0 = 1.0,z_0 = 0.,phi_0 = 0.,
                                                 directional_dependence = True)
# sample velocity
# get the R, z and phi colum
R,z,phi = location.T
#sample v interpolate
Rz_v = qdf.sampleV_interpolate(R, z, R_pixel = 0.01, z_pixel = 0.01)
R, z, vR, vT, vz = Rz_v.T
# convert to cartesian; this can be done since sampleV interpolate keeps the same
# original order, so phi order is in sync with R and 
coord_rect = cyl_to_rect(R, vR, vT, z, vz, phi)
# convert from natural to physics 
physical_rect = to_physical_units(coord_rect)

#get current date
date = time_class.localtime(time_class.time())[0:3]
#save data
np.save("qdf sample cartesian physical with new selection, e=1, date={}.npy".format(str(date)), physical_rect)

end = time_class.time()
print('time =', end - start)

