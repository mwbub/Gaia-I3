import sys
sys.path.append('..')

import numpy as np
from galpy.util.bovy_conversion import dens_in_msolpc3
from sampling.sampling import sample_location, sample_velocity

class toomredf:
    def __init__(self, n=1., ro=None, vo=None):
        """
        NAME:
            toomredf
            
        PURPOSE:
            an object implementing Toomre's 1982 distribution function
            
        INPUT:
            n - power of the df
            
            ro - reference distance from the GC
            
            vo - circular velocity at ro
            
        OUPUT:
            None
        """
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
        """
        NAME:
            turn_physical_on
            
        PURPOSE:
            activate input and output in physical units
            
        INPUT:
            None
            
        OUTPUT:
            None
        """
        self.use_physical = True
    
    def turn_physical_off(self):
        """
        NAME:
            turn_physical_off
            
        PURPOSE:
            deactivate input and output in physical units
            
        INPUT:
            None
            
        OUTPUT:
            None
        """
        self.use_physical = False
        
    def density(self, r, theta, use_physical=None):
        """
        NAME:
            density
            
        PURPOSE:
            evaluate the mass density of this DF at the spherical position
            (r, theta)
            
        INPUT:
            r - spherical radial position wrt the GC; natural units or kpc
            
            theta - angle measured from the z-axis; radians
            
            use_physical - boolean override of the current physical unit setting
            
        OUTPUT:
            mass density in natural units or solar masses per cubic parsec
        """
        if use_physical is None:
            use_physical = self.use_physical
            
        result = self._S(theta)/(4*np.pi*r**2)
        if use_physical:
            result *= self.ro**2 * dens_in_msolpc3(self.vo, self.ro)
            
        return result
    
    def density_cyl(self, R, z, use_physical=None):
        """
        NAME:
            density_cyl
            
        PURPOSE:
            evaluate the mass density of this DF at the cylindrical position
            (R, z)
            
        INPUT:
            R - cylindrical radial position wrt the GC; natural units or kpc
            
            z - cylindrical vertical position wrt the galactic plane; natural
            units or kpc
            
            use_physical - boolean override of the current physical unit setting
            
        OUTPUT:
            mass density in natural units or solar masses per cubic parsec    
        """
        r = np.sqrt(R**2 + z**2)
        theta = np.arcsin(R/r)
        return self.density(r, theta, use_physical=use_physical)
    
    def pvT(self, vT, use_physical=None):
        """
        NAME:
            pvT
            
        PURPOSE:
            evaluate the marginalized vT probability
            
        INPUT:
            vT - tangential velocity; natural units or km/s
            
            use_physical - boolean override of the current physical unit setting
            
        OUTPUT:
            p(vT)
        """
        if use_physical is None:
            use_physical = self.use_physical
        
        if use_physical:
            vT = vT/self.vo
            
        return vT**(2*self.n)*np.exp(-(self.n+1)*vT**2)
    
    def pvr(self, vr, use_physical=None):
        """
        NAME:
            pvr
            
        PURPOSE:
            evaluate the marginalized vr probability
            
        INPUT:
            vr - spherical radial velocity; natural units or km/s
            
            use_physical - boolean override of the current physical unit setting
            
        OUTPUT:
            p(vr)
        """
        if use_physical is None:
            use_physical = self.use_physical
            
        if use_physical:
            vr = vr/self.vo
            
        return np.exp(-(self.n+1)*vr**2)
    
    def pvtheta(self, vtheta, use_physical=None):
        """
        NAME:
            ptheta
            
        PURPOSE:
            evaluate the marginalized vtheta probability
            
        INPUT:
            vtheta - velocity in the direction of theta; natural units or km/s
            
            use_physical - boolean override of the current physical unit setting
            
        OUTPUT:
            p(vtheta)
        """
        return self.pvr(vtheta, use_physical=use_physical)
    
    
    def sampleV(self, size=1, use_physical=None):
        if use_physical is None:
            use_physical = self.use_physical
            
        sigma = 1/np.sqrt(2*self.n+2)
        maxvT = np.sqrt(self.n/(self.n+1))
        maxpvT = self.pvT(maxvT, use_physical=False)
        
        pvT = lambda v: self.pvT(v, use_physical=False)
        vT = sample_velocity(pvT, maxvT + 6*sigma, size, maxpvT)
        vr = np.random.normal(scale=sigma, size=size)
        vtheta = np.random.normal(scale=sigma, size=size)
        
        if use_physical:
            vT *= self.vo
            vr *= self.vo
            vtheta *= self.vo
            
        return np.stack((vT, vr, vtheta), axis=1)
    
    def samplePos(self, r_range, theta_range, phi_range, size=1, 
                  use_physical=None):
        if use_physical is None:
            use_physical = self.use_physical
            
        max_density = self.density(r_range[0], np.pi/2, use_physical=False)
        density = lambda r, theta: self.density(r, theta, use_physical=False)
        samples = sample_location(density, size, *r_range, *theta_range, 
                                  *phi_range, max_density)
        
        if use_physical:
            samples[:,0] *= self.ro
            
        return samples
    
    def _p(self, theta):
        return (1+np.cos(theta))**(self.n+1) + (1-np.cos(theta))**(self.n+1)
    
    def _S(self, theta):
        return 4*(self.n+1)*np.sin(theta)**(2*self.n)/self._p(theta)**2
    