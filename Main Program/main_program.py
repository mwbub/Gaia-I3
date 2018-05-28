"""
NAME:
    main_program

PURPOSE:
    


HISTORY:
    2018-05-28 - Written - Samuel Wong
"""
import os, sys
# get the outer folder as the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# import relevant functions from different folders
from search_phase_space.search_phase_space import search_phase_space
from check_uniformity_of_density.Integral_of_Motion import *
from check_uniformity_of_density.Linear_Algebra import *
from check_uniformity_of_density.Uniformity_Evaluation import *
from kde_function.kde_function import *

print(help(generate_KDE))