"""
NAME:
    qdf sample

PURPOSE:
    A function that samples location and veloicty under quasi-isothermal
    density function
    
HISTORY:
    2018-07-05 - Written - Samuel Wong
    2017-07-16 - Added sample velocity and use the interpolation sample location
"""
import sys
sys.path.append('../sampling')
from sampling import sample_location_interpolate
import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import time as time_class
#import qdf related things
from galpy.potential import MWPotential2014
from galpy.actionAngle import actionAngleAdiabatic
aA= actionAngleAdiabatic(pot=MWPotential2014,c=True)
# set up qdf
# qdf not imported from galpy since I am running a modified galpy in console
qdf= quasiisothermaldf(1./3.,0.2,0.1,1.,1.,pot=MWPotential2014,aA=aA,cutcounter=True)

#sample location
start = time_class.time()
# get the maximum of qdf
qdf_max = qdf.density(0.9375, 0)
# sample R from 7.5 kpc to 8.5 kpc; sample z from -0.5 kpc to 0.5 kpc
# let phi range from -arctan(0.5/8) to arctan(0.5/8)
location = sample_location_interpolate(qdf.density, 1700000, 0.9375, 1.0625,
                                       -0.0625, 0.0625, -0.0624, 0.0624, 
                                       df_max = qdf_max, pixel_R = 0.001,
                                       pixel_z = 0.001)
end = time_class.time()
np.save('qdf sample location', location)
print('time =', end - start)

# sample velocity
# get the R, z and phi colume
R = location[:, 0]
z = location[:, 1]
phi = location[:, 2]
# combine R and z into one array
Rz_set = np.stack((R, z), axis = 1)
#sample v on set
Rz_v = qdf.sampleV_on_set(Rz_set)
np.save('qdf sample Rzv', Rz_v)
# add back in phi coordinate
coord_v = np.dstack((Rz_v[:,0], Rz_v[:,1], phi, Rz_v[:, 2], Rz_v[:, 3], 
                     Rz_v[:, 4]))[0]
np.save('qdf sample data', coord_v)

    
    