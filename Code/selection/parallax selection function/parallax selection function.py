import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

# get data from fits file
data_g = Table.read("gaia_data_with_straight_cutoff.fits")
data_rv = Table.read("gaia_rv_with_straight_cutoff.fits")

# get rid of stars with negative parallax and stars further than 1 kpc
mask_g = np.all(np.array([data_g["parallax"]>0, 1/data_g["parallax"]<1]),axis = 0)
data_g = data_g[mask_g]
mask_rv = np.all(np.array([data_rv["parallax"]>0, 1/data_rv["parallax"]<1]),axis = 0)
data_rv = data_rv[mask_rv]

# get parallax
parallax_g = data_g["parallax"]
parallax_rv = data_rv["parallax"]

#set up min and max
parallax_min = 1
parallax_max = 10
#set up a list of even bins of 0.001 kpc in distance space, convert to uneven
#parralax bins
distance_pixel = 0.001
bins = np.flip(1/np.arange(1/parallax_max,1/parallax_min + distance_pixel, distance_pixel), axis =0)

#plot rv paralax distribution
plt.figure()
hist_parallax_rv, edges_rv, patches_rv = plt.hist(
        parallax_rv, range = (parallax_min, parallax_max), bins = bins)
plt.xlabel('Parallax')
plt.ylabel('Frequency')
plt.title("Parallax Distribution in RV")
plt.savefig("Parallax Distribution in RV.png")

#plot gaia paralax distribution
plt.figure()
hist_parallax_g, edges_g, patches_g = plt.hist(
        parallax_g, range = (parallax_min, parallax_max), bins = bins)
plt.xlabel('Parallax')
plt.ylabel('Frequency')
plt.title("Parallax Distribution in Gaia")
plt.savefig("Parallax Distribution in Gaia.png")

#plot ratio
plt.figure()
# since bins store the edge of each bin, it has one more point than the
# histogram. We want the x-array to be the center of each bin.
# Take every element except the first, subtract every element except the last.
# This gives an array of one less element representing the difference between
# each entry. Divide by 2 to find the distance between every two neighboring
# points. Add this back to every points except the last to get centralized
# points
bins_centralized = bins[:-1] + (bins[1:] - bins[:-1])/2
plt.plot(bins_centralized, hist_parallax_rv/hist_parallax_g)
plt.xlabel('Parallax')
plt.ylabel('Ratio (RV/G)')
plt.title("Ratio of Freqeuncy of Parallax")
plt.savefig("Ratio of Freqeuncy of Parallax.png")