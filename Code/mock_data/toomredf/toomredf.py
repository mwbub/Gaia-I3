import sys
sys.path.append('..')

import numpy as np
from galpy.util.bovy_conversion import dens_in_msolpc3
from galpy.potential import Potential
from sampling.sampling import sample_location, sample_velocity, \
    sample_location_selection

class toomredf:
    """
    Class implementing Toomre's 1982 distribution function
    """
    def __init__(self, n=1., ro=None, vo=None):
        """
        NAME:
            __init__
            
        PURPOSE:
            initialize a toomredf instance
            
        INPUT:
            n - power of the df
            
            ro - reference distance from the GC in kpc; turns ON physical input 
            and output if provided; default = 8 kpc
            
            vo - circular velocity at ro in km/s; turns ON physical input and 
            output if provided; default = 220 km/s
            
        OUPUT:
            None
        """
        self.n = n
        self.pot = ToomrePotential(n=n)
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
            
    def __call__(self, R, vR, vT, z, vz, use_physical=None):
        """
        NAME:
           __call__
           
        PURPOSE:
           return the DF
           
        INPUT:
            R, vR, vT, z, vz - galactocentric cylindrical coordinates; natural
            units or kpc and km/s
            
            use_physical - boolean override of the physical input/output setting

        OUTPUT:
           value of DF
        """
        if use_physical is None:
            use_physical = self.use_physical
            
        if use_physical:
            R = R/self.ro
            z = z/self.ro
            vR = vR/self.vo
            vT = vT/self.vo
            vz = vz/self.vo
            
        lz = R*vT
        E = 0.5*(vR**2 + vT**2 + vz**2) + self.pot(R,z)
        sigma2 = 1/(2*self.n+2)
        
        if use_physical:
            lz *= self.ro*self.vo
        
        return lz**(2*self.n)*np.exp(-E/sigma2)
    
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
            
            use_physical - boolean override of the physical input/output setting
            
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
            
            use_physical - boolean override of the physical input/output setting
            
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
            
            use_physical - boolean override of the physical input/output setting
            
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
            
            use_physical - boolean override of the physical input/output setting
            
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
            
            use_physical - boolean override of the physical input/output setting
            
        OUTPUT:
            p(vtheta)
        """
        return self.pvr(vtheta, use_physical=use_physical)
    
    def sampleV(self, size=1, use_physical=None):
        """
        NAME:
            sampleV
            
        PURPOSE:
            sample velocities in spherical coordinates
            
        INPUT:
            size - number of samples
            
            use_physical - boolean override of the physical input/output setting
            
        OUTPUT:
            vr, vtheta, vT in natural units or km/s
        """
        if use_physical is None:
            use_physical = self.use_physical
            
        sigma = 1/np.sqrt(2*self.n+2)
        maxvT = np.sqrt(self.n/(self.n+1))
        pvT = lambda v: self.pvT(v, use_physical=False)
        maxpvT = pvT(maxvT)

        vT = sample_velocity(pvT, maxvT + 6*sigma, size, maxpvT)
        vr = np.random.normal(scale=sigma, size=size)
        vtheta = np.random.normal(scale=sigma, size=size)
        
        if use_physical:
            vT *= self.vo
            vr *= self.vo
            vtheta *= self.vo
            
        return np.stack((vr, vtheta, vT), axis=1)
    
    def sampleV_cyl(self, size=1, use_physical=None):
        """
        NAME:
            sampleV_cyl
            
        PURPOSE:
            sample velocities in cylindrical coordinates
            
        INPUT:
            size - number of samples
            
            use_physical - boolean override of the physical input/output setting
            
        OUTPUT:
            vR, vz, vT in natural units or km/s
        """
        return self.sampleV(size=size, use_physical=use_physical)
    
    def samplePos(self, r_range, theta_range, phi_range, size=1, 
                  use_physical=None):
        """
        NAME:
            samplePos
            
        PURPOSE:
            sample positions in spherical coordinates
            
        INPUT:
            r_range - spherical radial range in which to sample stars; natural
            units or kpc
            
            theta_range - range of thetas in which to sample; radians
            
            phi_range - phi range over which to distribute the samples; radians
            
            size - number of samples
            
            use_physical - boolean override of the physical input/output setting
        
        OUTPUT:
            r, theta, phi
        """
        if use_physical is None:
            use_physical = self.use_physical
            
        if use_physical:
            r_range = [val/self.ro for val in r_range]
            
        density = lambda r, theta: self.density(r, theta, use_physical=False)
        max_density = density(r_range[0], np.pi/2)
        
        samples = sample_location(density, size, *r_range, *theta_range, 
                                  *phi_range, max_density)
        
        if use_physical:
            samples[:,0] *= self.ro
            
        return samples
    
    def samplePos_cyl(self, R_range, z_range, phi_range, size=1, 
                      use_physical=None):
        """
        NAME:
            samplePos_cyl
            
        PURPOSE:
            sample positions in cylindrical coordinates
            
        INPUT:
            R_range - cylindrical radial range in which to sample stars; natural
            units or kpc
            
            z_range - vertical range in which to sample; natural units or kpc
            
            phi_range - phi range over which to distribute the samples; radians
            
            size - number of samples
            
            use_physical - boolean override of the physical input/output setting
        
        OUTPUT:
            R, z, phi
        """
        if use_physical is None:
            use_physical = self.use_physical
            
        if use_physical:
            R_range = [val/self.ro for val in R_range]
            z_range = [val/self.ro for val in z_range]
            
        density = lambda R, z: self.density_cyl(R, z, use_physical=False)
        max_density = density(R_range[0], 0)
        
        samples = sample_location(density, size, *R_range, *z_range, *phi_range, 
                                  max_density)
        
        if use_physical:
            samples[:,:2] *= self.ro
            
        return samples
    
    def samplePos_cyl_selection(self, R_range, z_range, phi_range, selection,
                                size=1, use_physical=None, phi_0=np.pi, 
                                R_sun=8.3, dd=True):
        """
        NAME:
            samplePos_cyl_selection
            
        PURPOSE:
            sample positions in cylindrical coordinates with an additional
            selection function
            
        INPUT:
            R_range - cylindrical radial range in which to sample stars; natural
            units or kpc
            
            z_range - vertical range in which to sample; natural units or kpc
            
            phi_range - phi range over which to distribute the samples; radians
            
            seleciton - the selection function
            
            size - number of samples
            
            use_physical - boolean override of the physical input/output setting
            
            phi_0 - phi of the Sun
            
            R_sun - R position of the Sun in kpc
            
            dd - directional dependence
        
        OUTPUT:
            R, z, phi
        """
        if use_physical is None:
            use_physical = self.use_physical
            
        if use_physical:
            R_range = [val/self.ro for val in R_range]
            z_range = [val/self.ro for val in z_range]
            
        density = lambda R, z: self.density_cyl(R, z, use_physical=False)
        max_density = density(R_range[0], 0)
        
        samples = sample_location_selection(density, size, *R_range, *z_range, 
                                            *phi_range, max_density, selection,
                                            phi_0=phi_0, R_0=R_sun/8.,
                                            directional_dependence=dd)
        
        if use_physical:
            samples[:,:2] *= self.ro
            
        return samples
    
    def sample(self, r_range, theta_range, phi_range, size=1, 
               use_physical=None):
        """
        NAME:
            sample
            
        PURPOSE:
            sample positions and velocities in spherical coordinates
            
        INPUT:
            r_range - spherical radial range in which to sample stars; natural
            units or kpc
            
            theta_range - range of thetas in which to sample; radians
            
            phi_range - phi range over which to distribute the samples; radians
            
            size - number of samples
            
            use_physical - boolean override of the physical input/output setting
        
        OUTPUT:
            r, vr, vT, theta, vtheta, phi
        """
        vr, vtheta, vT = self.sampleV(size=size, use_physical=use_physical).T
        r, theta, phi = self.samplePos(r_range, theta_range, phi_range, 
                                       size=size, use_physical=use_physical).T
        return np.stack((r, vr, vT, theta, vtheta, phi), axis=1)
    
    def sample_cyl(self, R_range, z_range, phi_range, size=1,
                   use_physical=None):
        """
        NAME:
            sample_cyl
            
        PURPOSE:
            sample positions and velocities in cylindrical coordinates
            
        INPUT:
            R_range - cylindrical radial range in which to sample stars; natural
            units or kpc
            
            z_range - vertical range in which to sample; natural units or kpc
            
            phi_range - phi range over which to distribute the samples; radians
            
            size - number of samples
            
            use_physical - boolean override of the physical input/output setting
        
        OUTPUT:
            R, vR, vT, z, vz, phi
        """
        vR, vz, vT = self.sampleV_cyl(size=size, use_physical=use_physical).T
        R, z, phi = self.samplePos_cyl(R_range, z_range, phi_range, size=size,
                                       use_physical=use_physical).T
        return np.stack((R, vR, vT, z, vz, phi), axis=1)
    
    def sample_cyl_selection(self, R_range, z_range, phi_range, selection, 
                             size=1, use_physical=None):
        """
        NAME:
            sample_cyl
            
        PURPOSE:
            sample positions and velocities in cylindrical coordinates with an
            additional selection function
            
        INPUT:
            R_range - cylindrical radial range in which to sample stars; natural
            units or kpc
            
            z_range - vertical range in which to sample; natural units or kpc
            
            phi_range - phi range over which to distribute the samples; radians
            
            selection - the selection function
            
            size - number of samples
            
            use_physical - boolean override of the physical input/output setting
        
        OUTPUT:
            R, vR, vT, z, vz, phi
        """
        vR, vz, vT = self.sampleV_cyl(size=size, use_physical=use_physical).T
        R, z, phi = self.samplePos_cyl_selection(R_range, z_range, phi_range,
                                                 selection, size=size, 
                                                 use_physical=use_physical).T
        return np.stack((R, vR, vT, z, vz, phi), axis=1)
        
    def _p(self, theta):
        return (1+np.cos(theta))**(self.n+1) + (1-np.cos(theta))**(self.n+1)
    
    def _S(self, theta):
        return 4*(self.n+1)*np.sin(theta)**(2*self.n)/self._p(theta)**2
    
class ToomrePotential(Potential):
    """
    Potential for use with toomredf
    """
    def __init__(self, n=1., ro=None, vo=None):
        """
        NAME:
            __init__
            
        PURPOSE:
            initialize a ToomrePotential instance
            
        INPUT:
            n - power of the distribution function
            
            ro=, vo= distance and velocity scales for translation into internal 
            units (default from configuration file)
            
        OUTPUT:
            None
        """
        super().__init__(ro=ro, vo=vo)
        self.n = n
        
    def _evaluate(self, R, z, phi=0., t=0.):
        """
        NAME:
           _evaluate
           
        PURPOSE:
           evaluate the potential at R,z
           
        INPUT:
           R - Galactocentric cylindrical radius
           
           z - vertical height
           
           phi - azimuth
           
           t - time
           
        OUTPUT:
           Phi(R,z)
        """
        r = np.sqrt(R**2 + z**2)
        theta = np.arcsin(R/r)
        return np.log(r) + self._P(theta)
        
    def _P(self, theta):
        return np.log(self._p(theta)/2)/(self.n+1)
    
    def _p(self, theta):
        return (1+np.cos(theta))**(self.n+1) + (1-np.cos(theta))**(self.n+1)
