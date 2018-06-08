import numpy as np
import os, sys
# get the outer folder as the path
outer_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
check_uniformity_path =  os.path.abspath(os.path.join(outer_path, 'check_uniformity_of_density'))
sys.path.append(outer_path)
sys.path.append(check_uniformity_path)
# import relevant functions from different folders
from search_phase_space.search_phase_space import *
from check_uniformity_of_density.Integral_of_Motion import *
from check_uniformity_of_density.Linear_Algebra import *
from check_uniformity_of_density.Uniformity_Evaluation import *
from kde_function.kde_function import *
from tools.tools import *

total = 0
for i in range(1000):
    a = np.random.random(100)
    b = np.random.random(100)
    a = normalize_vector(a)[0]
    b = normalize_vector(b)[0]
    total += np.dot(a,b)
    
print(total/1000)