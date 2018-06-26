import sys
sys.path.append('../..')
import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D

data = np.load('dehnendf_MWPot.npz')
result = data['result']
cluster = data['cluster']

file_name = 'dehnendf_MWPot_epsilon=0.5'

# create graph of dot product
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
# get the maximum dot product at each cluster center
# change all nan to 0 in result for graphing purpose
result2 = np.nan_to_num(result)
max_dot_product2 = np.max(np.absolute(result2), axis = 1)
# scatter the cluster center x, y, and height is max dot product
ax.scatter(cluster[:, 0], cluster[:, 1], max_dot_product2, s = 1)
ax.set_title("Maximum Absolute Value of Dot Product in xy Dimension", fontsize=15)
ax.set_xlabel('x / 8 kpc')
ax.set_ylabel('y / 8 kpc')
ax.set_zlabel('maximum dot product')
# save figure
dot_product_figure_name = file_name + ', max dot product figure.pdf'
plt.savefig(dot_product_figure_name)
plt.show()