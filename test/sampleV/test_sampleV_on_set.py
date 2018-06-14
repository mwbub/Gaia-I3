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

#import qdf related things
from galpy.potential import MWPotential2014
from galpy.actionAngle import actionAngleAdiabatic
from galpy.df import quasiisothermaldf
aA= actionAngleAdiabatic(pot=MWPotential2014,c=True)

# set up qdf
qdf= quasiisothermaldf(1./3.,0.2,0.1,1.,1.,pot=MWPotential2014,aA=aA,cutcounter=True)

R_linspace = np.linspace(0, 15, 16)
z_linspace = np.linspace(-1, 1, 3)

Rv, zv = np.meshgrid(R_linspace, z_linspace)
Rv = Rv.reshape(-1,1)
zv = zv.reshape(-1,1)
data = np.concatenate((Rv, zv), axis = 1)

print(sampleV_on_set(data, qdf))