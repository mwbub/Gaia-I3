"""
NAME:
    main_program

PURPOSE:
    choose a point in phase space and check whether the density is changing
    locally in the four dimensional plane where energy and angular momentum
    are conserved. If it is not, I_3 does not exist.
    
HISTORY:
    2018-05-28 - Written - Samuel Wong, Mathew Bub
"""
import os, sys
# get the outer folder as the path
outer_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
check_uniformity_path =  os.path.abspath(os.path.join(outer_path, 'check_uniformity_of_density'))
sys.path.append(outer_path)
sys.path.append(check_uniformity_path)
# import relevant functions from different folders
from search_phase_space.search_phase_space import *
from check_uniformity_of_density.Integral_of_Motion import *
from check_uniformity_of_density.Linear_Algebra import *
from check_uniformity_of_density.Uniformity_Evaluation import *
from kde_function.kde_function import *
from astropy.coordinates import SkyCoord, CartesianRepresentation, CartesianDifferential
import astropy.units as unit


def galactic_to_galactocentric(point):
    u, v, w, U, V, W = point
    coord = SkyCoord(frame = 'galactic', representation_type = CartesianRepresentation,
                 differential_type = CartesianDifferential,
                 u = u*unit.kpc, v = v*unit.kpc, w = w*unit.kpc,
                 U = U*unit.km/unit.s, V = V*unit.km/unit.s, W = W*unit.km/unit.s)
    coord = coord.transform_to('galactocentric')
    coord.representation_type = CartesianRepresentation
    x = coord.x.value
    y = coord.y.value
    z = coord.z.value
    vx = coord.v_x.value
    vy = coord.v_y.value
    vz = coord.v_z.value
    return np.array([x, y, z, vx, vy, vz])


def galactocentric_to_galactic(point):
    x, y, z, vx, vy, vz = point
    coord = SkyCoord(frame = 'galactocentric', representation_type = CartesianRepresentation,
                 differential_type = CartesianDifferential,
                 x = x*unit.kpc, y = y*unit.kpc, z = z*unit.kpc,
                 v_x = vx*unit.km/unit.s, v_y = vy*unit.km/unit.s,
                 v_z = vz*unit.km/unit.s)
    coord = coord.transform_to('galactic')
    coord.representation_type = CartesianRepresentation
    u = coord.u.value
    v = coord.v.value
    w = coord.w.value
    U = coord.U.value
    V = coord.V.value
    W = coord.W.value
    return np.array([u, v, w, U, V, W])
    
# define parameters for the search and KDE
epsilon = 0.5
v_scale = 0.1
width = 10

# ask the user for input coordinate frame
frame = input("Do you want to search star in galactic or galactocentric coordinate? ")
if frame == "galactic":
    u  = float(input('u = '))
    v  = float(input('v = '))
    w  = float(input('w = '))
    U  = float(input('U = '))
    V  = float(input('V = '))
    W  = float(input('W = '))
    point_galactic = np.array([u, v, w, U, V, W])
    point_galactocentric = galactic_to_galactocentric(point_galactic)
elif frame == "galactocentric":
    x  = float(input('x = '))
    y  = float(input('y = '))
    z  = float(input('z = '))
    vx  = float(input('vx = '))
    vy = float(input('vy = '))
    vz  = float(input('vz = '))
    point_galactocentric = np.array([x, y, z, vx, vy, vz])
    point_galactic = galactocentric_to_galactic(point_galactocentric)
    
# get stars within an epsilon ball of the point in phase space from Gaia
# input the galactic coordinate into search function
table = search_phase_space(*point_galactic, epsilon, v_scale)
samples = table_to_samples(table)
print(samples)

# use the samples and a KDE learning method to generatea density function
density = generate_KDE(samples, 'gaussian', width)

# get the gradient of energy and momentum at the point
a = point_galactocentric # rename the galactocentric point to 'a'
del_E = grad(Energy, 6)
del_Lz = grad(L_z, 6)
del_E_a = del_E(a)
del_Lz_a = del_Lz(a)
# create matrix of the space spanned by direction of changing energy and momentum
V = np.array([del_E_a, del_Lz_a])
# get the 4 dimensional orthogonal complement of del E and del Lz
W = orthogonal_complement(V)
# evaluate if density is changing along the subspace 
# check to see if they are all 0; if so, it is not changing
directional_derivatives = evaluate_uniformity(density, a, W)
print('from a sample of {} of stars,'.format(np.shape(samples)[0]))
for i in range(len(directional_derivatives)):
    print('del_rho dot w_{} = {}'.format(i, directional_derivatives[i]))
