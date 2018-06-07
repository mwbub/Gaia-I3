"""
NAME:
    test_qdf
PURPOSE:
    Test the main function by replacing KDE with a quasiisothermal density
    function.
    
HISTORY:
    2018-06-01 - Written - Samuel Wong
"""
import os, sys
# get the outer folder as the path
outer_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(outer_path)
# import relevant functions from different folders
from main_program.main_program_single_star import *

#import qdf related things
from galpy.potential import MWPotential2014
from galpy.actionAngle import actionAngleAdiabatic
from galpy.df import quasiisothermaldf
aA= actionAngleAdiabatic(pot=MWPotential2014,c=True)

# set up qdf
qdf= quasiisothermaldf(1./3.,0.2,0.1,1.,1.,pot=MWPotential2014,aA=aA,cutcounter=True)

# define cartesian qdf
def cartesian_qdf(corrd):
    x, y, z, vx, vy, vz = corrd
    R, phi, z, vR, vT, vz = cartesian_to_cylindrical(x, y, z, vx, vy, vz)
    return qdf(R, vR, vT, z, vz)

main(cartesian_qdf)