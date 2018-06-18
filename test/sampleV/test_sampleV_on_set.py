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

# generate artificial data
R_linspace = np.linspace(0.5, 3., 20)
z_linspace = np.linspace(-2., 2., 15)
Rv, zv = np.meshgrid(R_linspace, z_linspace)
Rv = Rv.reshape(-1,1)
zv = zv.reshape(-1,1)
data = np.concatenate((Rv, zv), axis = 1)

# initialize a list that stores the result of real sampling multiple times
repeat = 5
real_result = []

# interpolate
start = time_class.time()
interpolated_result = sampleV_on_set(data, qdf)
inter_time = time_class.time() - start
# we take out the position and only compare velocity
interpolated_result = interpolated_result[:,2:]

# real sampling
start= time_class.time()
for j in range(repeat):
    real = np.empty(interpolated_result.shape)
    for (i,point) in enumerate(data):
        R, z = point
        vR, vT, vz = qdf.sampleV(R, z)[0]
        real[i] = vR, vT, vz
    real_result.append(real)
slow_time = time_class.time() - start
slow_time = slow_time/repeat
# find the mean of real result
real_result = np.array(real_result)
real_result = np.mean(real_result, axis = 0)

print('interpolation time = ', inter_time)
print('slow time = ', slow_time)

# we find the absolute value of the difference
result_difference = real_result - interpolated_result
# we get the array of fractional error and find the mean
error = np.mean(np.abs(result_difference[:, 1] / real_result[:, 1]))
print('fractional error in vT = ', error)

