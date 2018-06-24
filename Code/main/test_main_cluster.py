"""
NAME:
    test_main_cluster
PURPOSE:
    To run main program on mock_data 
    
HISTORY:
    2018-06-20 - Ran main program with QDF data - Michael Poon

"""
# import relevant functions from different folders
from main_program_cluster import * 
import numpy as np

samples = np.load('sampleV_at_(0.0,0.0,0.0)_epsilon=0.5.npy')

main(custom_samples = samples)