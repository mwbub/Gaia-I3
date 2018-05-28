from Linear_Algebra import *
from Uniformity_Evaluation import *
import numpy as np
from sympy import Matrix

def f(array):
    x = array[0]
    y = array[1]
    z = array[2]
    return 2*x*y + 3*y*z + z**2

# global variables
point = np.array([1., 1., 2.])
gradient = np.array([2., 8., 7.])
W = orthogonal_complement(np.array([gradient]))


def test_evaluate_uniformity_and_orthogonal_complement(f, point, W):
    # I hand calculated the gradient at the given point, ask orthogonal_complement function to calculate its orthogonal
    # space. Then ask evaluate_uniformit to find the dot product against the othogonal space
    # the answer should be 2 numbers very close to zero.
    # success implies that both orthogonal_complement and gradient have to work properly
    print('dot product between gradient and its orthonormal space:', evaluate_uniformity(f, point, W))


def test_orthonormality(W):
    # test whether the subspace output by orthogonal_complement are actually orthonormal
    # get dimensions of W
    m, n = np.shape(W)
    for i in range(m):
        print('norm of w[{}] = {}'.format(i, Matrix.norm(Matrix(W[i]))))

    for i in range(m):
        for j in range(m-1):
            if j != i:
                print('w[{}] dot w[{}] = {}'.format(i, j, np.dot(W[i],W[j])))

test_orthonormality(W)
print()
test_evaluate_uniformity_and_orthogonal_complement(f, point, W)
