import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
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
mask_g = np.all(np.array([distance_g>0,distance_g<10]), axis = 0)
ra_g = ra_g[mask_g]
dec_g = dec_g[mask_g]
distance_g =distance_g[mask_g]
mask_rv = np.all(np.array([distance_rv>0,distance_rv<10]), axis = 0)
ra_rv = ra_rv[mask_rv]
dec_rv = dec_rv[mask_rv]
distance_rv = distance_rv[mask_rv]

#set variables for histogram
ra_pixel = 1 # degree
dec_pixel = 1 # degree
distance_pixel = 0.001 # kpc
ra_min = min(np.min(ra_g), np.min(ra_rv))
dec_min = min(np.min(dec_g), np.min(dec_rv))
distance_min = min(np.min(distance_g), np.min(distance_rv))
ra_max = max(np.max(ra_g), np.max(ra_rv))
dec_max = max(np.max(dec_g), np.max(dec_rv))
distance_max = max(np.max(distance_g), np.max(distance_rv))
bin_ra = (ra_max - ra_min)/ra_pixel
bin_dec = (dec_max - dec_min)/dec_pixel
bin_distance = (distance_max - distance_min)/distance_pixel

# get 2d histogram for gaia
plt.figure(1)
histogram_g, xedges, yedges, graph = plt.hist2d(ra_g, dec_g, 
                                                bins = (bin_ra, bin_dec),
                                                range = [[ra_min, ra_max],
                                                         [dec_min, dec_max]])
plt.colorbar()
plt.xlabel('ra(deg)')
plt.ylabel('dec(deg)')
plt.title("Number of Stars in Gaia")
plt.savefig("Number of Stars in Gaia.png")

# get 2d histogram for rv
plt.figure(2)
histogram_rv, xedges, yedges, graph = plt.hist2d(ra_rv, dec_rv, 
                                                bins = (bin_ra, bin_dec),
                                                range = [[ra_min, ra_max],
                                                         [dec_min, dec_max]])
plt.colorbar()
plt.xlabel('ra(deg)')
plt.ylabel('dec(deg)')
plt.title("Number of Stars in RV")
plt.savefig("Number of Stars in RV.png")

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
