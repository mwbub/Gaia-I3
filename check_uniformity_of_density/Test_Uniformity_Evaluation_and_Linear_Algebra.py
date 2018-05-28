from Linear_Algebra import *
from Uniformity_Evaluation import *
import numpy as np

def f(array):
    x = array[0]
    y = array[1]
    z = array[2]
    return 2*x*y + 3*y*z + z**2

def test_evaluate_uniformity_and_orthogonal_complement():
    # I hand calculate the gradient at the given point, ask orthogonal_complement function to calculate its orthogonal
    # space. Then ask evaluate_uniformit to find the dot product against the othogonal space
    # the answer should be 2 numbers very close to zero.
    # success implies that both orthogonal_complement and gradient have to work properly
    point = np.array([1., 1., 2.])
    gradient = np.array([2., 8., 7.])
    W = orthogonal_complement(np.array([gradient]))
    print(evaluate_uniformity(f, point, W))

