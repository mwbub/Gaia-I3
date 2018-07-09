import numpy as np
from sample_location import sample_location
import pylab as plt

def df(R,z):
    return np.abs(R*z)

Rzset = sample_location(df,1000, -1, 1, -1, 1, 0, 1)
R = Rzset[:,0]
z = Rzset[:,1]
plt.scatter(R,z)