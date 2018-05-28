from Linear_Algebra import *
from Uniformity_Evaluation import *
import numpy as np

def f(array):
    x = array[0]
    y = array[1]
    z = array[2]
    return 2*x*y + 3*y*z + z**2

point = np.array([1., 1., 2.])
gradient = np.array([2., 8., 7.])
W = orthogonal_complement(np.array([gradient]))
print(evaluate_uniformity(f, point, W))

