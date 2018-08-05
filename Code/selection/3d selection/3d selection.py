import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy import units
from astropy.coordinates import SkyCoord
from matplotlib.colors import LogNorm

# get data from fits file
data_g = Table.read("gaia_data_adjusted_for_rv.fits")
data_rv = Table.read("gaia_intersection_with_rv.fits")
# get coordinates of both gaia and rv
ra_g = data_g['ra']
dec_g = data_g['dec']
parallax_g = data_g['parallax']
ra_rv = data_rv['ra']
dec_rv = data_rv['dec']
parallax_rv = data_rv['parallax']
# get distance
distance_g = 1/parallax_g # in kpc
distance_rv = 1/parallax_rv # in kpc

# get rid of stars with negative distance and distance further than 10kpc
mask_g = np.all(np.array([distance_g>0,distance_g<1]), axis = 0)
ra_g = ra_g[mask_g]
dec_g = dec_g[mask_g]
distance_g = distance_g[mask_g]
mask_rv = np.all(np.array([distance_rv>0,distance_rv<1]), axis = 0)
ra_rv = ra_rv[mask_rv]
dec_rv = dec_rv[mask_rv]
distance_rv = distance_rv[mask_rv]

# put both gaia and rv into sky coord and convert to cartesian
g_icrs = SkyCoord(ra = ra_g*units.deg,
                   dec = dec_g*units.deg,
                   distance = distance_g*units.kpc)
g_galcen = g_icrs.transform_to('galactocentric')
g_galcen.representation_type = 'cartesian'
g_coord = np.array([g_galcen.x.value, g_galcen.y.value, g_galcen.z.value]).T

rv_icrs = SkyCoord(ra = ra_rv*units.deg,
                   dec = dec_rv*units.deg,
                   distance = distance_rv*units.kpc)
rv_galcen = rv_icrs.transform_to('galactocentric')
rv_galcen.representation_type = 'cartesian'
rv_coord = np.array([rv_galcen.x.value, rv_galcen.y.value, rv_galcen.z.value]).T


#set variables for histogram
pixel = 0.001 # kpc
x_min = min(np.min(rv_coord[:,0]), np.min(g_coord[:,0]))
y_min = min(np.min(rv_coord[:,1]), np.min(g_coord[:,1]))
z_min = min(np.min(rv_coord[:,2]), np.min(g_coord[:,2]))
x_max = max(np.max(rv_coord[:,0]), np.max(g_coord[:,0]))
y_max = max(np.max(rv_coord[:,1]), np.max(g_coord[:,1]))
z_max = max(np.max(rv_coord[:,2]), np.max(g_coord[:,2]))
bin_x = (x_max - x_min)/pixel
bin_y = (y_max - y_min)/pixel
bin_z = (z_max - z_min)/pixel

# get 3d histogram for gaia
histogram_g, edges_g = np.histogramdd(g_coord, bins = (bin_x, bin_y, bin_z))
# get 3d histogram for rv
histogram_rv, edges_rv = np.histogramdd(rv_coord, bins = (bin_x, bin_y, bin_z))

# define number of stars as a function of position
def number(x, y, z, histogram):
    x_index = ((x-x_min)/pixel).astype(int)-1
    y_index = ((y-y_min)/pixel).astype(int)-1
    z_index = ((z-z_min)/pixel).astype(int)-1
    if np.ndim(x) == 1:
        result = []
        for i in range(np.size(x_index)):
            result.append(histogram[x_index[i]][y_index[i]][z_index[i]])
        return np.array(result)

# define the ratio of number density as a function of position
def ratio(x, y, z):
    return number(x, y, z, histogram_rv)/number(x, y, z, histogram_g)

"""
# compute an array of ratio and plot
x_linspace = np.linspace(x_min, x_max, bin_x)
y_linspace = np.linspace(y_min, y_max, bin_y)
x_v, y_v = np.meshgrid(x_linspace, y_linspace)
result = ratio(x_v, y_v)
# change nan to zero
result = np.nan_to_num(result)
plt.figure()
plt.pcolor(ra_v, dec_v, z, norm = LogNorm())
plt.colorbar()
plt.xlabel('ra(deg)')
plt.ylabel('dec(deg)')
plt.title("Ratio of Number in RV/Gaia")
plt.savefig("Ratio of Number in RV and Gaia.png")
"""