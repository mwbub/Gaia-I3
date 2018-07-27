import numpy as np
import matplotlib.pyplot as plt

# get ra dec coordinate of both gaia and rv
ra_g, dec_g = np.linspace(1.,100.,100), np.linspace(1.,100.,100)
ra_rv, dec_rv = np.linspace(1.,100.,100), np.linspace(1.,100.,100)
#set variables for histogra
ra_pixel = 0.1
dec_pixel = 0.1
ra_min = min(np.min(ra_g), np.min(ra_rv))
dec_min = min(np.min(dec_g), np.min(dec_rv))
ra_max = max(np.max(ra_g), np.max(ra_rv))
dec_max = max(np.max(dec_g), np.max(dec_rv))
bin_ra = (ra_max - ra_min)/ra_pixel
bin_dec = (dec_max - dec_min)/dec_pixel

# get 2d histogram for gaia
histogram_g, xedges, yedges, graph = plt.hist2d(ra_g, dec_g, 
                                                bins = (bin_ra, bin_dec),
                                                range = [[ra_min, ra_max],
                                                         [dec_min, dec_max]])
plt.colorbar()
plt.xlabel('ra(mas)')
plt.ylabel('dec(mas)')
plt.title("Density of Gaia in ra and dec")
plt.savefig("Density of Gaia in ra and dec.png")

# get 2d histogram for rv
histogram_rv, xedges, yedges, graph = plt.hist2d(ra_rv, dec_rv, 
                                                bins = (bin_ra, bin_dec),
                                                range = [[ra_min, ra_max],
                                                         [dec_min, dec_max]])
plt.colorbar()
plt.xlabel('ra(mas)')
plt.ylabel('dec(mas)')
plt.title("Density of RV in ra and dec")
plt.savefig("Density of Gaia in ra and dec.png")

# define number density as a function of ra and dec
def number_density(ra, dec, histogram):
    ra_index = ((ra-ra_min)/ra_pixel).astype(int)-1
    dec_index = ((dec-dec_min)/dec_pixel).astype(int)-1
    result = []
    for i in range(np.size(ra_index)):
        result.append(histogram[ra_index[i]][dec_index[i]])
    return np.array(result)

# define the ratio of number density as a function of ra and dec
def ratio(ra, dec):
    return number_density(ra, dec, histogram_rv)/number_density(ra, dec, histogram_g)

# compute an array of ratio
ra_linspace = np.linspace(ra_min, ra_max, 100)
dec_linspace = np.linspace(dec_min, dec_max, 100)
ra_v, dec_v = np.meshgrid(ra_linspace, dec_linspace)
z = ratio(ra_v, dec_v)
plt.pcolor(ra_v, dec_v, z)
