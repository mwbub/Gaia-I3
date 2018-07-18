"""
Copy and paste this file into a folder containing the results of the main 
program to make the new plots for that data set (L_z-E projection and a 
histogram of the dot products).

If you're not lazy like me, you might consider adding these plots to the main
program and/or rewriting this file so you don't have to copy and paste.
"""
import sys
sys.path.append('../../..') # this line might break depending on how many levels
                            # deep your folder is
import numpy as np
import matplotlib.pyplot as plt
from check_uniformity_of_density.Integral_of_Motion import Energy, L_z

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