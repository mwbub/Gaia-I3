"""
NAME:
    test_sampleV_on_set
PURPOSE:
    Test sampleV on set
    
HISTORY:
    2018-06-13 - Written - Samuel Wong
"""
import numpy as np
from sampleV_on_set import sampleV_on_set
import time as time_class

#import qdf related things
from galpy.potential import MWPotential2014
from galpy.actionAngle import actionAngleAdiabatic
from galpy.df import quasiisothermaldf
aA= actionAngleAdiabatic(pot=MWPotential2014,c=True)

# set up qdf
qdf= quasiisothermaldf(1./3.,0.2,0.1,1.,1.,pot=MWPotential2014,aA=aA,cutcounter=True)

R_linspace = np.linspace(1, 14, 13)
z_linspace = np.linspace(-4, 5, 10)

Rv, zv = np.meshgrid(R_linspace, z_linspace)
Rv = Rv.reshape(-1,1)
zv = zv.reshape(-1,1)
data = np.concatenate((Rv, zv), axis = 1)

start= time_class.time()
interpolated_result = sampleV_on_set(data, qdf)
inter_time = time_class.time() - start

real_result = np.empty(interpolated_result.shape)
start= time_class.time()
for (i,point) in enumerate(data):
    R, z = point
    print('about to sample ', R, z)
    vR, vT, vz = qdf.sampleV(R, z)[0]
    real_result[i] = vR, vT, vz
    print('finished ', i)
slow_time = time_class.time() - start

print('interpolation time = ', inter_time)
print('slow time = ', slow_time)

# we take out the position and only compare the real and interpolated velocity
interpolated_result = interpolated_result[:,2:]
# we find the absolute value of the difference
result_difference = real_result - interpolated_result
# we get the array of fractional error and find the mean
error = np.mean(np.abs(result_difference / real_result))
print('fractional error = ', error)

