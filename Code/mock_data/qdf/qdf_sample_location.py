"""
NAME:
    qdf sample location

PURPOSE:
    A function that samples location under quasi-isothermal density function

FUNCTIONS:
    
HISTORY:
    2018-07-05 - Written - Samuel Wong
"""
import sys
sys.path.append('..')
from sample_location import sample_location
import numpy as np
import pylab as plt
#import qdf related things
from galpy.potential import MWPotential2014
from galpy.actionAngle import actionAngleAdiabatic
from galpy.df import quasiisothermaldf
aA= actionAngleAdiabatic(pot=MWPotential2014,c=True)
# set up qdf
qdf= quasiisothermaldf(1./3.,0.2,0.1,1.,1.,pot=MWPotential2014,aA=aA,cutcounter=True)

location = sample_location(qdf.density,100, 0, 1, -1, 1, 0, np.pi/2)
R = location[:, 0]
z = location[:, 1]
phi = location[:, 2]

fig = plt.figure(figsize=(8, 8), facecolor='black')
plt.style.use("dark_background")
ax = fig.add_subplot(111, projection='3d')
ax.scatter(R*np.cos(phi), R*np.sin(phi), z, s = 10)
    
    