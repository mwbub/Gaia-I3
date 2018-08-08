import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

# get data from fits file
data_g = Table.read("gaia_data_with_straight_cutoff.fits")
data_rv = Table.read("gaia_rv_with_straight_cutoff.fits")

# get rid of stars with negative parallax
mask_g = data_g["parallax"]>0
data_g = data_g[mask_g]
mask_rv = data_rv["parallax"]>0
data_rv = data_rv[mask_rv]

# get parallax
parallax_g = data_g["parallax"]
parallax_rv = data_rv["parallax"]

#set up min and max
parallax_min = min(np.min(parallax_g), np.min(parallax_rv))
parallax_max = min(max(np.max(parallax_g), np.max(parallax_rv)),10)

bins = 1000

#plot rv paralax distribution
plt.figure()
hist_parallax_rv, edges_rv, patches_rv = plt.hist(
        parallax_rv, range = (parallax_min, parallax_max), bins = bins)
plt.xlabel('Parallax')
plt.ylabel('Frequency')
plt.title("Parallax Distribution in RV (equal bin width)")
plt.savefig("Parallax Distribution in RV (equal bin width).png")

#plot gaia paralax distribution
plt.figure()
hist_parallax_g, edges_g, patches_g = plt.hist(
        parallax_g, range = (parallax_min, parallax_max), bins = bins)
plt.xlabel('Parallax')
plt.ylabel('Frequency')
plt.title("Parallax Distribution in Gaia (equal bin width)")
plt.savefig("Parallax Distribution in Gaia (equal bin width).png")

#plot ratio
plt.figure()
plt.plot(np.linspace(parallax_min, parallax_max, bins), hist_parallax_rv/hist_parallax_g)
plt.xlabel('Parallax')
plt.ylabel('Ratio (RV/G)')
plt.title("Ratio of Freqeuncy of Parallax (equal bin width)")
plt.savefig("Ratio of Freqeuncy of Parallax (equal bin width).png")