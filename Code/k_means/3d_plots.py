import sys
sys.path.append('..')

from search import search_local
from sklearn.cluster import MiniBatchKMeans, KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

v_scale = 0.1
cluster_scale = v_scale*0.5
epsilon = 0.8
samples = search_local.search_phase_space(0, 0, 0, 0, 0, 0, epsilon, v_scale)
scaled_samples = np.concatenate((samples[:,:3], samples[:,3:]*cluster_scale),
                                axis=1)

n_clusters = int(input('n_clusters = '))
kmeans = KMeans(n_clusters=n_clusters, max_iter=1000)
kmeans.fit(scaled_samples)

cmap = plt.get_cmap('jet')
colours = cmap(np.linspace(0, 1, n_clusters))

fig = plt.figure(figsize=(18, 8))

pos_ax = fig.add_subplot(121, projection='3d')
pos_ax.scatter(*samples.transpose()[:3], s=5, c=colours[kmeans.labels_])
pos_ax.scatter(*kmeans.cluster_centers_.transpose()[:3], c='black',
               marker='x', s=100)
pos_ax.set_title('Positions')
pos_ax.set_xlabel('$x$ (kpc)')
pos_ax.set_ylabel('$y$ (kpc)')
pos_ax.set_zlabel('$z$ (kpc)')

vel_ax = fig.add_subplot(122, projection='3d')
vel_ax.scatter(*samples.transpose()[3:], s=5, c=colours[kmeans.labels_])
vel_ax.scatter(*kmeans.cluster_centers_.transpose()[3:]/cluster_scale, 
               c='black', marker='x', s=100)
vel_ax.set_title('Velocities')
vel_ax.set_xlabel('$v_x$ (km/s)')
vel_ax.set_ylabel('$v_y$ (km/s)')
vel_ax.set_zlabel('$v_z$ (km/s)')

plt.show()
