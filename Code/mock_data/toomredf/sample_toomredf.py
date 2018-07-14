import sys
sys.path.append('..')

from toomredf import toomredf
from sampling.sampling import sample_location

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm

df = toomredf(n=4.)
minR = 0.1
maxR = 3
minz = -2
maxz = 2
maxrho = df.density_cyl(minR, 0)
