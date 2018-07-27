"""
NAME:
    test_fake_density

PURPOSE:
    Test the main function by replacing KDE with a fake density function
    
HISTORY:
    2018-05-31 - Written - Samuel Wong
"""
import numpy as np
import os, sys
# get the outer folder as the path
outer_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(outer_path)
# import relevant functions from different folders
from main.main_program_cluster import *
data = np.load("data.npz")
cluster = data["cluster"]
print(np.shape(cluster))

def fake_density(coord):
    e = Energy(coord)
    l = L_z(coord)
    return e**2 + l**2

main(uniformity_method = "projection", gradient_method = "numeric",
         search_method = "local", custom_density = fake_density, custom_samples = None,
          custom_centres=cluster)