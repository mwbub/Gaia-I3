import sys
sys.path.append('../../../..')
sys.path.append('../../..')

import numpy as np
import matplotlib.pyplot as plt
from check_uniformity_of_density.Integral_of_Motion_toomre import Energy, L_z

data = np.load('data.npz')
cluster_centres = data['cluster']

dot_products = data['result']
max_dot_products = np.nanmax(np.abs(dot_products), axis=1)

cluster_centres = cluster_centres[~np.isnan(max_dot_products)]
max_dot_products = max_dot_products[~np.isnan(max_dot_products)]

energy = Energy(cluster_centres.T)
angular_momentum = L_z(cluster_centres.T)

with plt.style.context(('dark_background')):
    plt.figure(figsize=(10,8), facecolor='black')
    plt.scatter(angular_momentum, energy, c=max_dot_products, marker='.', s=5,
                cmap='plasma', vmin=0, vmax=1)
    plt.colorbar(label='Maximum Absolute Dot Product')
    plt.xlabel('$L_z$')
    plt.ylabel('$E$')
    plt.title('Maximum Absolute Value of Dot Product in $L_z-E$ Dimension')
    plt.savefig('color dot product L_z-E figure.png')
    plt.show()

plt.figure(figsize=(10,8))
plt.hist(max_dot_products, bins=30, density=True)
plt.xlabel('Maximum Absolute Dot Product')
plt.ylabel('Frequency')
plt.title('Maximum Absolute Values of the Dot Products')
plt.savefig('dot product histogram.png')
plt.show()