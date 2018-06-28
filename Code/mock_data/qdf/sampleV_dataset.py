import os, sys
outer_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(outer_path)
from check_uniformity_of_density.Integral_of_Motion import *
from search import search_local
from tools.tools import *
from galpy.util.bovy_coords import rect_to_cyl
from galpy.df import quasiisothermaldf
from galpy.actionAngle import actionAngleStaeckel
import astropy.units as u

aAS= actionAngleStaeckel(pot=MWPotential2014,delta=0.45,c=True)
qdfS= quasiisothermaldf(1./3.,0.2,0.1,1.,1.,pot=MWPotential2014,aA=aAS,cutcounter=True)

epsilon = 0.5
point_galactocentric, point_galactic = get_star_coord_from_user()
samples = search_local.search_phase_space(*point_galactic, epsilon, 0)
R, phi, z = rect_to_cyl(*samples.T[:3])

R /= 8.
z /= 8.

vR = np.empty(len(samples))
vT = np.empty(len(samples))
vz = np.empty(len(samples))
for i in range(len(samples)):
    s = qdfS.sampleV(R[i], z[i], n=1)
    vR[i] = s[0, 0]
    vT[i] = s[0, 1]
    vz[i] = s[0, 2]
    print('{:.2f}% complete'.format(100*i/(len(samples)-1)))

R *= 8.
z *= 8.
vR *= 220.
vT *= 220.
vz *= 220.

sampleV_set = np.stack([R, vR, vT, z, vz, phi], axis=1)

if not os.path.exists('data'):
    os.mkdir('data')
filename = 'data/sampleV_at_({},{},{})_epsilon={}'.format(*point_galactic[:3], epsilon)
np.save(filename, sampleV_set)