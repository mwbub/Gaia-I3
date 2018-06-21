"""
NAME:
    temp
PURPOSE:

    
HISTORY:

"""
# import relevant functions from different folders
from main.main_program_cluster import *
import numpy as np
samples = np.load('temp.npy')

main(custom_samples = samples)