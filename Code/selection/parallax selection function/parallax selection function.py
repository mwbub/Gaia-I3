import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

bins = 100

# get data from fits file
data_g = Table.read("gaia_data_with_straight_cutoff.fits")
data_rv = Table.read("gaia_rv_with_straight_cutoff.fits")

# get rid of stars with negative parallax and further than 30 kpc
mask_g = np.all(np.array([data_g["parallax"]>0, 1/data_g["parallax"]<30]),
                axis = 0)
data_g = data_g[mask_g]
mask_rv = np.all(np.array([data_rv["parallax"]>0, 1/data_rv["parallax"]<30]),
                 axis =0)
data_rv = data_rv[mask_rv]

# get parallax
parallax_g = data_g["parallax"]
parallax_rv = data_rv["parallax"]

parallax_min = min(np.min(parallax_g), np.min(parallax_rv))
# ignore everything closer than 1/5 = 0.2 kpc
parallax_max = min(max(np.max(parallax_g), np.max(parallax_rv)), 3)
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
plt.plot(np.linspace(parallax_min, parallax_max, bins),
         hist_parallax_rv/hist_parallax_g)
plt.xlabel('Parallax')
plt.ylabel('Ratio (RV/G)')
plt.title("Ratio of Freqeuncy of Parallax")
plt.savefig("Ratio of Freqeuncy of Parallax.png")