import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

# get data from fits file
data_g = Table.read("gaia_data_adjusted_for_rv.fits")
data_rv = Table.read("gaia_intersection_with_rv.fits")
# get ra dec coordinate of both gaia and rv
ra_g = data_g['ra']
dec_g = data_g['dec']
ra_rv = data_rv['ra']
dec_rv = data_rv['dec']
#set variables for histogram
ra_pixel = 1 # degree
dec_pixel = 1 # degree
ra_min = min(np.min(ra_g), np.min(ra_rv))
dec_min = min(np.min(dec_g), np.min(dec_rv))
ra_max = max(np.max(ra_g), np.max(ra_rv))
dec_max = max(np.max(dec_g), np.max(dec_rv))
bin_ra = (ra_max - ra_min)/ra_pixel
bin_dec = (dec_max - dec_min)/dec_pixel

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
plt.figure(3)
plt.pcolor(ra_v, dec_v, z)
plt.colorbar()
plt.xlabel('ra(deg)')
plt.ylabel('dec(deg)')
plt.title("Ratio of Number in RV/Gaia")
plt.savefig("Ratio of Number in RV and Gaiac.png")
