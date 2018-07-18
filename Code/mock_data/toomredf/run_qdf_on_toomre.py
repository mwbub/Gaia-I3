import sys
sys.path.append('../..')
sys.path.append('../../check_uniformity_of_density')

from main.main_program_cluster import main
from sample_toomredf import sample_like_gaia
import numpy as np

from galpy.potential import MWPotential2014
from galpy.actionAngle import actionAngleAdiabatic
from galpy.df import quasiisothermaldf
aA = actionAngleAdiabatic(pot=MWPotential2014,c=True)

from tools.tools import rect_to_cyl

qdf = quasiisothermaldf(1./3.,0.2,0.1,1.,1.,pot=MWPotential2014,aA=aA,cutcounter=True)

def df(x):
    return qdf(*rect_to_cyl(*x).T[:-1])[0]

centres = np.load('main_program_results/toomre_main/data.npz')['cluster']
print('getting data')
data = sample_like_gaia(4,4)
print('running main')
main(custom_samples=data, custom_density=df, custom_centres=centres,
     gradient_method='analytic')