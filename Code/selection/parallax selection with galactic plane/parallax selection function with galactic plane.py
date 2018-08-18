import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.coordinates import SkyCoord
from scipy.interpolate import UnivariateSpline
import dill
dill.settings['recurse'] = True # make dill chase down dependency

#set up min and max
parallax_min = 1
parallax_max = 10
# define galactic width
plane_width = 10. #degree

# get data from fits file
data_g = Table.read("gaia_data_with_straight_cutoff.fits")
data_rv = Table.read("gaia_rv_with_straight_cutoff.fits")
# get rid of stars with negative parallax
mask_g = data_g["parallax"]>0
data_g = data_g[mask_g]
mask_rv = data_rv["parallax"]>0
data_rv = data_rv[mask_rv]
# get coordinates of both gaia and rv
ra_g = data_g['ra'] #degree
dec_g = data_g['dec'] #degree
parallax_g = data_g["parallax"]
ra_rv = data_rv['ra'] #degree
dec_rv = data_rv['dec'] #degree
parallax_rv = data_rv["parallax"]
# convert ra and dec to galactic coordinate
galactic_g = SkyCoord(ra_g, dec_g, unit="deg", frame="icrs").galactic
l_g = galactic_g.l.degree
b_g = galactic_g.b.degree
galactic_rv = SkyCoord(ra_rv, dec_rv, unit="deg", frame="icrs").galactic
l_rv = galactic_rv.l.degree
b_rv = galactic_rv.b.degree
#Divide all gaia data into galactic plane and elsewhere
g_galactic_mask = np.abs(b_g) < plane_width
l_g_galactic = l_g[g_galactic_mask]
b_g_galactic = b_g[g_galactic_mask]
parallax_g_galactic = parallax_g[g_galactic_mask]
l_g_elsewhere = l_g[~g_galactic_mask]
b_g_elsewhere = b_g[~g_galactic_mask]
parallax_g_elsewhere = parallax_g[~g_galactic_mask]
rv_galactic_mask = np.abs(b_rv) < plane_width
l_rv_galactic = l_rv[rv_galactic_mask]
b_rv_galactic = b_rv[rv_galactic_mask]
parallax_rv_galactic = parallax_rv[rv_galactic_mask]
l_rv_elsewhere = l_rv[~rv_galactic_mask]
b_rv_elsewhere = b_rv[~rv_galactic_mask]
parallax_rv_elsewhere = parallax_rv[~rv_galactic_mask]

#find parallax ratio in glactic plane

#set up a list of even bins of 0.001 kpc in distance space, convert to uneven
#parralax bins
distance_pixel = 0.001
bins = np.flip(1/np.arange(1/parallax_max,1/parallax_min + distance_pixel, distance_pixel), axis =0)
bins_centralized = bins[:-1] + (bins[1:] - bins[:-1])/2
#plot rv paralax distribution
plt.figure()
hist_parallax_rv_galactic, edges_rv, patches_rv = plt.hist(
    parallax_rv_galactic, range = (parallax_min, parallax_max), bins = bins)
plt.xlabel('Parallax')
plt.ylabel('Frequency')
plt.title("Galactic Plane Parallax Distribution in RV, parallax=({},{})".format(parallax_min, parallax_max))
#plot gaia paralax distribution
plt.figure()
hist_parallax_g_galactic, edges_g, patches_g = plt.hist(
        parallax_g_galactic, range = (parallax_min, parallax_max), bins = bins)
plt.xlabel('Parallax')
plt.ylabel('Frequency')
plt.title("Galactic Plane Parallax Distribution in Gaia, parallax=({},{})".format(parallax_min, parallax_max))
#plot ratio
plt.figure()
ratio_galactic = hist_parallax_rv_galactic/hist_parallax_g_galactic
plt.plot(bins_centralized, ratio_galactic)
plt.xlabel('Parallax')
plt.ylabel('Ratio (RV/G)')
plt.title("Galactic Plane Ratio of Freqeuncy of Parallax, parallax=({},{})".format(parallax_min, parallax_max))

#fit ratio galactic
spl_galactic = UnivariateSpline(bins_centralized, ratio_galactic)
# plot the fitted function for galactic
xs = np.linspace(1,10, 1000)
plt.figure()
plt.plot(bins_centralized, ratio_galactic, color = 'b', label = "Data")
plt.plot(xs, spl_galactic(xs), color = 'r', label = "Interpolation")
plt.xlabel('Parallax')
plt.ylabel('Ratio (RV/G)')
plt.legend()
plt.title("Ratio of Freqeuncy of Parallax in Galactic Plane (Fitted)")
plt.savefig("Ratio of Freqeuncy of Parallax in Galactic Plane (Fitted).png")

# define a wrapper function for galactic selection
def selection_galactic_plane(parallax):
    mask_high = parallax >= 10
    mask_low = parallax <= 1
    result = spl_galactic(parallax)
    result[mask_high] = 0.7
    result[mask_low] = 0.
    return result
# plot the extrapolated selection function for galactic
xs = np.linspace(0,15, 1000)
plt.figure()
plt.plot(bins_centralized, ratio_galactic, color = 'b', label = "Data")
plt.plot(xs, selection_galactic_plane(xs), color = 'r', label = "Interpolation")
plt.xlabel('Parallax')
plt.ylabel('Ratio (RV/G)')
plt.legend()
plt.title("Ratio of Freqeuncy of Parallax in Galactic Plane (Extrapolated)")
plt.savefig("Ratio of Freqeuncy of Parallax in Galactic Plane (Extrapolated).png")


#find ratio of rv to galactic elsewhere

#plot rv paralax distribution
plt.figure()
hist_parallax_rv_elsewhere, edges_rv, patches_rv = plt.hist(
    parallax_rv_elsewhere, range = (parallax_min, parallax_max), bins = bins)
plt.xlabel('Parallax')
plt.ylabel('Frequency')
plt.title("Elsewhere Plane Parallax Distribution in RV, parallax=({},{})".format(parallax_min, parallax_max))
#plot gaia paralax distribution
plt.figure()
hist_parallax_g_elsewhere, edges_g, patches_g = plt.hist(
        parallax_g_elsewhere, range = (parallax_min, parallax_max), bins = bins)
plt.xlabel('Parallax')
plt.ylabel('Frequency')
plt.title("Elsewhere Plane Parallax Distribution in Gaia, parallax=({},{})".format(parallax_min, parallax_max))
#plot ratio for elsewhere
plt.figure()
bins_centralized = bins[:-1] + (bins[1:] - bins[:-1])/2
ratio_elsewhere = hist_parallax_rv_elsewhere/hist_parallax_g_elsewhere
plt.plot(bins_centralized, ratio_elsewhere)
plt.xlabel('Parallax')
plt.ylabel('Ratio (RV/G)')
plt.title("Elsewhere Plane Ratio of Freqeuncy of Parallax, parallax=({},{})".format(parallax_min, parallax_max))

#fit ratio elsewhere
spl_elsewhere = UnivariateSpline(bins_centralized, ratio_elsewhere)
# plot the fitted function for elsewhere
xs = np.linspace(1,10, 1000)
plt.figure()
plt.plot(bins_centralized, ratio_elsewhere, color = 'b', label = "Data")
plt.plot(xs, spl_elsewhere(xs), color = 'r', label = "Interpolation")
plt.xlabel('Parallax')
plt.ylabel('Ratio (RV/G)')
plt.legend()
plt.title("Ratio of Freqeuncy of Parallax elsewhere (Fitted)")
plt.savefig("Ratio of Freqeuncy of Parallax elsewhere (Fitted).png")

# define a wrapper function for elsewhere selection
def selection_elsewhere(parallax):
    mask_high = parallax >= 10
    mask_low = parallax <= 1
    result = spl_elsewhere(parallax)
    result[mask_high] = 0.79
    result[mask_low] = 0.
    return result
# plot the extrapolated selection function for elsewhere
xs = np.linspace(0,15, 1000)
plt.figure()
plt.plot(bins_centralized, ratio_elsewhere, color = 'b', label = "Data")
plt.plot(xs, selection_elsewhere(xs), color = 'r', label = "Interpolation")
plt.xlabel('Parallax')
plt.ylabel('Ratio (RV/G)')
plt.legend()
plt.title("Ratio of Freqeuncy of Parallax elsewhere (Extrapolated)")
plt.savefig("Ratio of Freqeuncy of Parallax elsewhere (Extrapolated).png")


#define the overall selection function
def selection(parallax, l, b):
    mask = np.abs(b) < plane_width
    out = np.empty(np.shape(b)[0])
    out[mask] = selection_galactic_plane(parallax[mask])
    out[~mask] = selection_elsewhere(parallax[~mask])
    return out


# save the function object
with open("selection_function", "wb") as dill_file:
    dill.dump(selection, dill_file)

# code needed to retrieve the function
#with open("selection_function", "rb") as dill_file:
#    selection = dill.load(dill_file)
    
#plot overall selection on both galactic plane and elsewhere as test
plt.figure()
plt.plot(xs, selection(xs, np.zeros(1000), np.linspace(-10.,10.,1000)), color = "blue", label="galactic")
plt.plot(xs, selection(xs, np.zeros(1000), np.linspace(10.,30.,1000)), color = "green", label="elsewhere")
plt.legend()
plt.xlabel('Parallax')
plt.ylabel('selection')