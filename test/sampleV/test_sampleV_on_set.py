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
from scipy.stats import ks_2samp

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

# number of repeat to average out vT
repeat = 10

# interpolate
start = time_class.time()
interpolated_result = sampleV_on_set(data, qdf)
inter_time = time_class.time() - start
# we take out the position and only compare velocity
interpolated_result = interpolated_result[:,2:]

# real sampling
# initialize a list that stores the result of real sampling multiple times
real_result = np.empty(interpolated_result.shape)
start= time_class.time()
for (i,point) in enumerate(data):
    R, z = point
    vR = np.empty(repeat)
    vT = np.empty(repeat)
    vz = np.empty(repeat)
    for j in range(repeat):
        vR[j], vT[j], vz[j] = qdf.sampleV(R, z)[0]
    real_result[i] = vR[0], np.mean(vT), vz[0]
slow_time = time_class.time() - start
slow_time = slow_time/repeat

# we find the absolute value of the difference
result_difference = real_result - interpolated_result
# we get the array of fractional error and find the mean
error_vT = np.mean(np.abs(result_difference[:, 1] / real_result[:, 1]))

# we check whether the interpolated vR and real vR are from the same distribution
# by ks test; same for vz
vR_ks = ks_2samp(real_result[:, 0], interpolated_result[:,0])
vz_ks = ks_2samp(real_result[:, 2], interpolated_result[:,2])

print('interpolation time = ', inter_time)
print('slow time = ', slow_time)
print('fractional error in vT = ', error_vT)
print('vR ks statistic = ', vR_ks)
print('vz ks statistic = ', vz_ks)

np.save("real and interpolated result, repeat both are 10", real_result, interpolated_result)