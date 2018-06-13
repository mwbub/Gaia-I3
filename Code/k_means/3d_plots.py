import sys
sys.path.append('..')

from search import search_local
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import iqr
import numpy as np

def plot_points(title, points, cluster_centres, n_clusters, cluster_labels):
    cmap = plt.get_cmap('jet')
    colours = cmap(np.linspace(0, 1, n_clusters))
               
    fig = plt.figure(figsize=(18, 8))
    fig.suptitle(title, fontsize=16)

    pos_ax = fig.add_subplot(121, projection='3d')
    pos_ax.scatter(*points[:3], s=5, c=colours[cluster_labels])
    pos_ax.scatter(*cluster_centres[:3], s=100, c='black', marker='x')
    pos_ax.set_title('Positions')
    pos_ax.set_xlabel('$x$ (kpc)')
    pos_ax.set_ylabel('$y$ (kpc)')
    pos_ax.set_zlabel('$z$ (kpc)')

    vel_ax = fig.add_subplot(122, projection='3d')
    vel_ax.scatter(*points[3:], s=5, c=colours[cluster_labels])
    vel_ax.scatter(*cluster_centres[3:], s=100, c='black', marker='x')
    vel_ax.set_title('Velocities')
    vel_ax.set_xlabel('$v_x$ (km/s)')
    vel_ax.set_ylabel('$v_y$ (km/s)')
    vel_ax.set_zlabel('$v_z$ (km/s)')
    
    fig.subplots_adjust(wspace=0)
               
    plt.show()

def plot_clusters(points, n_clusters, cluster_scale):
    kmeans = KMeans(n_clusters=n_clusters, max_iter=1000)
    kmeans.fit(np.concatenate((points[:,:3], points[:,3:]*cluster_scale), axis=1))
    
    cluster_centres = np.concatenate((kmeans.cluster_centers_[:,:3], 
                                      kmeans.cluster_centers_[:,3:]/cluster_scale), axis=1)
    
    plot_points('cluster_scale = {:.2f}'.format(cluster_scale),
                points.transpose(),
                cluster_centres.transpose(),
                n_clusters,
                kmeans.labels_)
    
def plot_with_iqr_scale(points, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, max_iter=1000)
    points_iqr = iqr(points, axis=0, nan_policy='omit')
    kmeans.fit(points/points_iqr)
    
    plot_points('Scaled for IQR',
                points.transpose(),
                (kmeans.cluster_centers_*points_iqr).transpose(),
                n_clusters,
                kmeans.labels_)
    
def plot_with_std_scale(points, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, max_iter=1000)
    points_std = np.nanstd(points, axis=0)
    kmeans.fit(points/points_std)
    
    plot_points('Scaled for $\sigma$',
                points.transpose(),
                (kmeans.cluster_centers_*points_std).transpose(),
                n_clusters,
                kmeans.labels_)
    
epsilon = float(input('epsilon = '))
v_scale = float(input('v_scale = '))
n_clusters = int(input('n_clusters = '))

scale_type = None
while scale_type not in ['a','b','c']:
    scale_type = input('choose scale type - manual (a), std (b) or iqr (c): ')
    
samples = search_local.search_phase_space(0, 0, 0, 0, 0, 0, epsilon, v_scale)    
    
if scale_type == 'a':
    cluster_scale = float(input('cluster_scale = '))
    plot_clusters(samples, n_clusters, cluster_scale)
elif scale_type == 'b':
    plot_with_std_scale(samples, n_clusters)
else:
    plot_with_iqr_scale(samples, n_clusters)
