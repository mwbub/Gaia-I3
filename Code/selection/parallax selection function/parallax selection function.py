import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from scipy.interpolate import UnivariateSpline
import dill
dill.settings['recurse'] = True # make dill chase down dependency

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
ratio = hist_parallax_rv/hist_parallax_g
plt.plot(bins_centralized, ratio)
plt.xlabel('Parallax')
plt.ylabel('Ratio (RV/G)')
plt.title("Ratio of Freqeuncy of Parallax")
plt.savefig("Ratio of Freqeuncy of Parallax.png")

# fit the selection ratio with univariate spline
spl = UnivariateSpline(bins_centralized, ratio)

# plot the fitted function
xs = np.linspace(1,10, 1000)
plt.figure()
plt.plot(bins_centralized, ratio, color = 'b', label = "Data")
plt.plot(xs, spl(xs), color = 'r', label = "Interpolation")
plt.xlabel('Parallax')
plt.ylabel('Ratio (RV/G)')
plt.legend()
plt.title("Ratio of Freqeuncy of Parallax (Fitted)")
plt.savefig("Ratio of Freqeuncy of Parallax (Fitted).png")

# define a wrapper function for selection
def selection(parallax):
    mask_high = parallax >= 10
    mask_low = parallax <= 1
    result = spl(parallax)
    result[mask_high] = 0.77
    result[mask_low] = 0.
    return result

# plot the extrapolated selection function
xs = np.linspace(0,15, 1000)
plt.figure()
plt.plot(bins_centralized, ratio, color = 'b', label = "Data")
plt.plot(xs, selection(xs), color = 'r', label = "Interpolation")
plt.xlabel('Parallax')
plt.ylabel('Ratio (RV/G)')
plt.legend()
plt.title("Ratio of Freqeuncy of Parallax (Extrapolated)")
plt.savefig("Ratio of Freqeuncy of Parallax (Extrapolated).png")

# save the function object
with open("selection_function", "wb") as dill_file:
    dill.dump(selection, dill_file)

# code needed to retrieve the function
#with open("selection_function", "rb") as dill_file:
#    selection = dill.load(dill_file)