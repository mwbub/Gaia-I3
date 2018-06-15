import numpy as np
import astropy.units as u
from galpy.df import dehnendf
from galpy.orbit import Orbit
from galpy.potential import MWPotential2014

def get_3d_samples(n=1, rrange=None):
    df = dehnendf()
    sampled_ROrbits = df.sample(n=n, rrange=rrange)
    
    R = np.array([o.R() for o in sampled_ROrbits])
    vRZ = np.array([o.vR() for o in sampled_ROrbits])
    vT = np.array([o.vT() for o in sampled_ROrbits])
    
    kRZ = vRZ**2/2
    kR = kRZ * np.random.random(n)
    kz = kRZ - kR
    vR = np.sqrt(2*kR)
    vz = np.sqrt(2*kz)
    
    vxvv = [[R[i], vR[i], vT[i], 0, vz[i]] for i in range(n)]
    orbits = [Orbit(vxvv=vxvv[i]) for i in range(n)]
    
    ts = np.linspace(0, 3, 1000) * u.Gyr
    for o in orbits:
        o.integrate(ts, MWPotential2014)
    
    return orbits