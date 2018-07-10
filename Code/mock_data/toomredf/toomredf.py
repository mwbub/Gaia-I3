import numpy as np

class toomredf:
    def __init__(self, n=1, amp=1., r1=1., ro=8., vo=220.):
        self.n = n
        self.amp = amp
        self.r1 = r1
        self.ro = ro
        self.vo = vo
        
    def _p(self, theta):
        return (1+np.cos(theta))**(self.n+1) + (1-np.cos(theta))**(self.n+1)
    
    def _S(self, theta):
            return 4*(self.n+1)*np.sin(theta)**(2*self.n)/self._p(theta)**2
        
    def pRtheta(self, r, theta, physical=False):
        if physical:
            r /= self.ro
        return self.amp*(self.r1/r)**2*self._S(theta)
    
    def density(self, R, z, physical=False):
        r = np.sqrt(R**2 + z**2)
        theta = np.arcsin(R/r)
        return self.pRtheta(r, theta, physical=physical)