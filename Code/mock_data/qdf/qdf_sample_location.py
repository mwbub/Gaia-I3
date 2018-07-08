"""
NAME:
    qdf sample location

PURPOSE:
    A function that samples location under quasi-isothermal density function

FUNCTIONS:
    
HISTORY:
    2018-07-05 - Written - Samuel Wong
"""
import numpy as np
from scipy import integrate
#import qdf related things
from galpy.potential import MWPotential2014
from galpy.actionAngle import actionAngleAdiabatic
from galpy.df import quasiisothermaldf
aA= actionAngleAdiabatic(pot=MWPotential2014,c=True)
# set up qdf
qdf= quasiisothermaldf(1./3.,0.2,0.1,1.,1.,pot=MWPotential2014,aA=aA,cutcounter=True)

    
    
    