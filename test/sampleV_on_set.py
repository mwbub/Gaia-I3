"""
NAME:
    sampleV_on_set
PURPOSE:
    Sample velocity of a set of points in a density function by using
    interpolation of a grid, thereby improving the efficiency of sampleV.
    
HISTORY:
    2018-06-11 - Written - Samuel Wong
"""
#import qdf related things
from galpy.potential import MWPotential2014
from galpy.actionAngle import actionAngleAdiabatic
from galpy.df import quasiisothermaldf
aA= actionAngleAdiabatic(pot=MWPotential2014,c=True)

# set up qdf
qdf= quasiisothermaldf(1./3.,0.2,0.1,1.,1.,pot=MWPotential2014,aA=aA,cutcounter=True)

def sampleV_on_set(rz_set, df):
    """
    NAME:
        sampleV_on_set

    PURPOSE:
        Given a three dimensional density function (df), as well as a set of 
        r and z coordinates of stars, return three sampled velocity for each
        star.

    INPUT:
        rz_set = a numpy array containing a list of (r,z) coordinate; assumed
                 to be all in natural unit
        df = a galpy three dimensional density function

    OUTPUT:
        coordinate_v = a numpy array containing the original coordinate but
                        with velocity attached. Each coordinate is of the form
                        (r, z, vR, vT, vz)

    HISTORY:
        2018-06-11 - Written - Samuel Wong
    """