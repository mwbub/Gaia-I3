import numpy as np
from sampling import *
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D

def df1(R,z):
    return 2*R**3

def df2(R,z):
    return 3*R*z**2

def vdf1(v):
    return (1/np.sqrt(2*np.pi*(0.2**2)))*np.exp(-((v-40)**2)/(2*(0.2**2)))

def test_sample_location(df, n, R_min, R_max, z_min, z_max, phi_min, phi_max,
                         df_max, std_R, std_z):
    location = sample_location(df, n, R_min, R_max, z_min, z_max, phi_min,
                               phi_max, df_max)
    R = location[:, 0]
    z = location[:, 1]
    phi = location[:, 2]
    
    fig = plt.figure(figsize=(8, 8), facecolor='black')
    plt.style.use("dark_background")
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(R*np.cos(phi), R*np.sin(phi), z, s = 10)
    plt.show()

    print('theoretical std of R =', std_R)
    print('std of R = ', np.std(R))
    print('theoretical std of z =', std_z)
    print('std of z = ', np.std(z))

def test_sample_velocity(vdf, v_max, n, vdf_max, std):
    v = sample_velocity(vdf, v_max, n, vdf_max)
    
    print('theoretical std of v =', std)
    print('experimental std of v=', np.std(v))
    
    plt.figure()
    plt.style.use("dark_background")
    plt.hist(v)
    plt.show()
    
def test_sample_location_interpolate(df, n, R_min, R_max, z_min, z_max,
                                     phi_min, phi_max, df_max, std_R, std_z):
    location = sample_location_interpolate(df, n, R_min, R_max, z_min, z_max,
                                           phi_min, phi_max, df_max, 0.01, 0.01)
    R = location[:, 0]
    z = location[:, 1]
    phi = location[:, 2]
    
    """
    fig = plt.figure(figsize=(8, 8), facecolor='black')
    plt.style.use("dark_background")
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(R*np.cos(phi), R*np.sin(phi), z, s = 10)
    plt.show()
    """

    print('theoretical std of R =', std_R)
    print('std of R = ', np.std(R))
    print('theoretical std of z =', std_z)
    print('std of z = ', np.std(z))
    
test_sample_location(df1,1000000, 0, 1, -1, 1, 0, 2*np.pi, 2, 0.163, 0.577)
test_sample_location(df2, 1000000, 0, 1, -1, 1, 0, 2*np.pi, 3, 0.236, 0.775)
test_sample_velocity(vdf = vdf1, v_max = 80, n= 100000, vdf_max= 1.99, std =0.2)
test_sample_location_interpolate(df2, 1000000, 0, 1, -1, 1, 0, 2*np.pi, 3, 0.236, 0.775)
    