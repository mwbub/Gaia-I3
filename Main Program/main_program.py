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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# import relevant functions from different folders
from search_phase_space.search_phase_space import *
from check_uniformity_of_density.Integral_of_Motion import *
from check_uniformity_of_density.Linear_Algebra import *
from check_uniformity_of_density.Uniformity_Evaluation import *
from kde_function.kde_function import *
from astropy.coordinates import SkyCoord, CartesianRepresentation, CartesianDifferential
import astropy.units as unit

# define parameters for the search and KDE
epsilon = 0.5
v_scale = 0
width = 1

# get stars within an epsilon ball of a star in phase space from Gaia
# start with galactic coordinate
u = -7.7e+00*unit.kpc
v = 1.1e+00*unit.kpc
w = 1.2e-01*unit.kpc
U = 2.5e+01*unit.km/unit.s
V = 2.3e+02*unit.km/unit.s
W = -5.8e-01*unit.km/unit.s
# get the nearby star data
table = search_phase_space(u,v,w,U,V,W,epsilon, v_scale)
samples = table_to_samples(table)
print(samples)

# use the samples and a KDE learning method to generatea density function
density = generate_KDE(samples, 'gaussian', width)

# convert the central star to galactocentric coordinate
coord = SkyCoord(frame = 'galactic', representation_type = CartesianRepresentation,
                 differential_type = CartesianDifferential,
                 u = u, v = v, w = w, U = U, V = V, W = W)
coord = coord.transform_to('galactocentric')
x = coord.x.value
y = coord.y.value
z = coord.z.value
vx = coord.v_x.value
vy = coord.v_y.value
vz = coord.v_z.value
# define phase space point
a = np.array([x,y,z,vx,vy,vz])

# get the gradient of energy and momentum at the point
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
