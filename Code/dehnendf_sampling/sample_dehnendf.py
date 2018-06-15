import numpy as np
import astropy.units as u
from galpy.df import dehnendf
from galpy.orbit import Orbit
from galpy.potential import MWPotential2014

def get_3d_samples(n=1, rrange=None, integration_steps=100):
    df = dehnendf()
    sampled_ROrbits = df.sample(n=n, rrange=rrange)
    
    R = np.array([o.R() for o in sampled_ROrbits])
    vRz = np.array([o.vR() for o in sampled_ROrbits])
    vT = np.array([o.vT() for o in sampled_ROrbits])
    
    kRz = vRz**2/2
    kR = kRz * np.random.random(n)
    kz = kRz - kR
    vR = np.sqrt(2*kR) * np.sign(vRz)
    vz = np.sqrt(2*kz) * np.random.choice((1, -1), size=n)
    
    vxvv = [[R[i], vR[i], vT[i], 0, vz[i]] for i in range(n)]
    orbits = [Orbit(vxvv=vxvv[i]) for i in range(n)]
    
    t = np.linspace(0, 500, integration_steps) * u.Myr
    for o in orbits:
        o.integrate(t, MWPotential2014)
    
    return orbits