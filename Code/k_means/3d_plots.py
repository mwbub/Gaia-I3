import sys
sys.path.append('..')

from search import search_local
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_clusters(points, n_clusters, cluster_scale):
    kmeans = KMeans(n_clusters=n_clusters, max_iter=1000)
    kmeans.fit(np.concatenate((points[:,:3], points[:,3:]*cluster_scale), axis=1))
               
    cmap = plt.get_cmap('jet')
    colours = cmap(np.linspace(0, 1, n_clusters))
               
    fig = plt.figure(figsize=(18, 8))
    fig.suptitle('cluster_scale = {:.2f}'.format(cluster_scale), fontsize=16)

    pos_ax = fig.add_subplot(121, projection='3d')
    pos_ax.scatter(*points.transpose()[:3], s=5, c=colours[kmeans.labels_])
    pos_ax.scatter(*kmeans.cluster_centers_.transpose()[:3], s=100,
                   c='black', marker='x')
    pos_ax.set_title('Positions')
    pos_ax.set_xlabel('$x$ (kpc)')
    pos_ax.set_ylabel('$y$ (kpc)')
    pos_ax.set_zlabel('$z$ (kpc)')

    vel_ax = fig.add_subplot(122, projection='3d')
    vel_ax.scatter(*points.transpose()[3:], s=5, c=colours[kmeans.labels_])
    vel_ax.scatter(*kmeans.cluster_centers_.transpose()[3:]/cluster_scale, 
                   s=100, c='black', marker='x')
    vel_ax.set_title('Velocities')
    vel_ax.set_xlabel('$v_x$ (km/s)')
    vel_ax.set_ylabel('$v_y$ (km/s)')
    vel_ax.set_zlabel('$v_z$ (km/s)')
    
    fig.subplots_adjust(wspace=0)
               
    plt.show()
    
epsilon = float(input('epsilon = '))
v_scale = float(input('v_scale = '))
n_cluster = int(input('n_cluster = '))
cluster_scale = float(input('cluster_scale = '))
samples = search_local.search_phase_space(0, 0, 0, 0, 0, 0, epsilon, v_scale)
plot_clusters(samples, n_cluster, cluster_scale)
