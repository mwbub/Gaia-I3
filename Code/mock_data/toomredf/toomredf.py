import numpy as np
from galpy.util.bovy_conversion import dens_in_msolpc3

class toomredf:
    def __init__(self, n=1., ro=None, vo=None):
        self.n = n
        self.use_physical = False
        
        if ro is None:
            self.ro = 8.
        else:
            self.ro = ro
            self.use_physical = True
            
        if vo is None:
            self.vo = 220.
        else:
            self.vo = vo
            self.use_physical = True
    
    def turn_physical_on(self):
        self.use_physical = True
    
    def turn_physical_off(self):
        self.use_physical = False
        
    def density(self, r, theta, use_physical=None):
        if use_physical is None:
            use_physical = self.use_physical
            
        result = self._S(theta)/(4*np.pi*r**2)
        if use_physical:
            result *= self.ro**2 * dens_in_msolpc3(self.vo, self.ro)
            
        return result
    
    def density_cyl(self, R, z, use_physical=None):
        r = R**2 + z**2
        theta = np.arcsin(R/r)
        return self.density(self, r, theta, use_physical=use_physical)
    
    def pvphi(self, vphi, use_physical=None):
        if use_physical is None:
            use_physical = self.use_physical
        
        if use_physical:
            vphi = vphi/self.vo
            
        return vphi**(2*self.n)*np.exp(-(self.n+1)*vphi**2)
    
    def pvr(self, vr, use_physical=None):
        if use_physical is None:
            use_physical = self.use_physical
            
        if use_physical:
            vr = vr/self.vo
            
        return np.exp(-(self.n+1)*vr**2)
    
    def pvtheta(self, vtheta, use_physical=None):
        return self.p_vr(vtheta, use_physical=use_physical)
    
    def _p(self, theta):
        return (1+np.cos(theta))**(self.n+1) + (1-np.cos(theta))**(self.n+1)
    
    def _S(self, theta):
        return 4*(self.n+1)*np.sin(theta)**(2*self.n)/self._p(theta)**2
    