import numpy as np
from sample_location import sample_location
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D

def df(R,z):
    return R**3

location = sample_location(df,1000, 0, 1, -1, 1, 0, 2*np.pi)
R = location[:, 0]
z = location[:, 1]
phi = location[:, 2]
print(np.shape(location))

fig = plt.figure(figsize=(8, 8), facecolor='black')
plt.style.use("dark_background")
ax = fig.add_subplot(111, projection='3d')
ax.scatter(R*np.cos(phi), R*np.sin(phi), z, s = 10)