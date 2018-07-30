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
bin_x = (x_max - x_min)/x_pixel
bin_y = (y_max - y_min)/y_pixel
bin_z = (z_max - z_min)/z_pixel

# get 3d histogram for gaia
gaia_coord = np.array([ra_g, dec_g, distance_g]).T #(N,3) array of coordinates
histogram_g, edges_g = np.histogramdd(gaia_coord, bins = (bin_ra, bin_dec, bin_distance))

# get 2d histogram for rv
rv_coord = np.array([ra_rv, dec_rv, distance_rv]).T #(N,3) array of coordinates
histogram_rv, edges_rv = np.histogramdd(rv_coord, bins = (bin_ra, bin_dec, bin_distance))

"""
# define number of stars as a function of ra and dec
def number(ra, dec, histogram):
    ra_index = ((ra-ra_min)/ra_pixel).astype(int)-1
    dec_index = ((dec-dec_min)/dec_pixel).astype(int)-1
    if np.ndim(ra) == 1:
        result = []
        for i in range(np.size(ra_index)):
            result.append(histogram[ra_index[i]][dec_index[i]])
        return np.array(result)
    elif np.ndim(ra) == 2:
        result = []
        for i in range(np.shape(ra_index)[0]):
            row = []
            for j in range(np.shape(ra_index)[1]):
                row.append(histogram[ra_index[i][j]][dec_index[i][j]])
            result.append(row)
        return np.array(result)

# define the ratio of number density as a function of ra and dec
def ratio(ra, dec):
    return number(ra, dec, histogram_rv)/number(ra, dec, histogram_g)

# compute an array of ratio and plot
ra_linspace = np.linspace(ra_min, ra_max, bin_ra)
dec_linspace = np.linspace(dec_min, dec_max, bin_dec)
ra_v, dec_v = np.meshgrid(ra_linspace, dec_linspace)
z = ratio(ra_v, dec_v)
# change nan to zero
z = np.nan_to_num(z)
plt.figure(3)
plt.pcolor(ra_v, dec_v, z, norm = LogNorm())
plt.colorbar()
plt.xlabel('ra(deg)')
plt.ylabel('dec(deg)')
plt.title("Ratio of Number in RV/Gaia")
plt.savefig("Ratio of Number in RV and Gaia.png")
"""