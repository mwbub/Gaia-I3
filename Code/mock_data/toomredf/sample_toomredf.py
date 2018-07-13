import sys
sys.path.append('..')

from toomredf import toomredf
from sampling.sampling import sample_location, sample_velocity
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

df = toomredf(n=4., ro=8., vo=220)
minR = 0.1 * 8
maxR = 4 * 8
minz = -4 * 8
maxz = 4 * 8
maxrho = df.density_cyl(minR, 0)
pos_samples = sample_location(df.density_cyl, 100000, minR, maxR, minz, maxz, 0, 0)

cmap = cm.get_cmap('inferno')
cmap.set_bad(cmap(0))
plt.hist2d(*pos_samples.T[:2], bins=50, norm=colors.LogNorm(), cmap=cmap)
plt.colorbar()
plt.show()