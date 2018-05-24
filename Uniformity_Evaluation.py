"""
    NAME:
        Uniformity_Evaluation

    PURPOSE:
        This module contains a function that evaluates whether a given function is uniform along a given subspace.
        In addition, it contains functions that return gradient and partial derivative of a given function.

    FUNCTIONS:
        evaluate_uniformity: This function takes a differeniable function, a point, and the vectors generating
                             a subspace, and return whether the function is uniform along the subspace at that point.

        gradient: This function takes a differentiable function and returns the gradient of that function

        partial_derivative: This function takes a function and the order of the input variable with respect to which
                            the partial derivative is to be taken. Then it returns the partial derivative with respect
                            to that variable.

    HISTORY:
        2018-05-24 - Written - Samuel Wong
    """
import numpy as np
from scipy.misc import derivative


def evaluate_uniformity(f, x, W):
    """
    NAME:
        evaluate_uniformity

    PURPOSE:
        Given a differentiable function, f, a n-dimensional vector, x, and a m dimensional subscpace of R^n
        (m <= n) that is spanned by the vectors contained in W, return the directional derivative of f along
        each of the vector. If all of them are close to zero, then f is said to be uniform in the subspace.

    INPUT:
        f = a differentiable function that takes n number to 1 number
        n = number of variables that f takes
        W = a numpy array that contains m number of n-dimensional linearly independent vectors (warning: requries m<n);
            Each vector is also a numpy array

    OUTPUT:
        directional_derivatives = a numpy array containing the directional derivative of f along each direction
                                  of the basis vectors generating the subspace

    HISTORY:
        2018-05-24 - Written - Samuel Wong
    """
    # get the size of the subspace
    m = np.size(W)
    # get the size of the domain spae
    n = np.size(x)
    # get the gradient of f
    del_f = gradient(f, n)
    # evaluate the gradient at the given point
    del_f_x = del_f(x)
    # initialize directional_derivatives, where each component is gradient dotted with one of the vectors
    directional_derivatives = np.empty(m)
    for i in range(m):
        directional_derivatives[i] = np.dot(del_f_x, W[i])
    return directional_derivatives
