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
