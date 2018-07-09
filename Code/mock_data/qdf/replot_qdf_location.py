import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D

location = np.load('qdf sample location.npy')
R = location[:, 0]
z = location[:, 1]
phi = location[:, 2]

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(R*np.cos(phi), R*np.sin(phi), z, s = 1)
plt.savefig('10000 qdf sample location.png')