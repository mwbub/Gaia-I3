import numpy as np
from sampling import sample_location, sample_velocity
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D

def df(R,z):
    return R**3

def vdf(v):
    return np.exp(-(v-1)**2/0.2)

def test_sample_location():
    location = sample_location(df,1000, 0, 1, -1, 1, 0, 2*np.pi)
    R = location[:, 0]
    z = location[:, 1]
    phi = location[:, 2]
    print(np.shape(location))
    
    fig = plt.figure(figsize=(8, 8), facecolor='black')
    plt.style.use("dark_background")
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(R*np.cos(phi), R*np.sin(phi), z, s = 10)

def test_sample_velocity():
    v = sample_velocity(vdf, 3, 1000)
    plt.style.use("dark_background")
    plt.hist(v)
    
test_sample_velocity()