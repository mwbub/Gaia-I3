import numpy as np
from scipy.misc import derivative
from Linear_Algebra import *
from numpy import linalg as LA

def partial_derivative(f, i):
    """
    NAME:
        partial_derivative

    PURPOSE:
        Given an n-dimensional real-valued function, f, return the ith partial derivative function of f.

    INPUT:
        f = a differentiable function that takes a numpy array of n numbers to 1 number
        i = the position of the variable in the input array with respect to which we want to take the partial derivative

    OUTPUT:
        evaluate_partial_derivative = the ith partial derivative function of f.
                                      This function is also an n-dimensional real-valued function

    HISTORY:
        2018-05-24 - Written - Samuel Wong
    """
    # define a function capable of evaluating the partial derivative at a n-dimensional point
    def evaluate_partial_derivative(point):

        # define a function that treats all variables as constant except for the ith input
        def fixed_value_except_ith(x_i):
            new_point = np.concatenate((point[:i], [x_i], point[i+1:]))
            return f(new_point)

        # now that we have the function treating all other variables as constants except for the ith one, we have a
        # normal 1 dimensional derivative; evaluate it at the ith coordinate of the point
        # set dx to sufficiently small number
        normal_derivative = derivative(fixed_value_except_ith, point[i], dx=1e-8)
        return normal_derivative

    # return the function that can evaluate partial derivative at a point, without giving it any input.
    # so this is the partial derivative function
    return evaluate_partial_derivative


def grad(f, n):
    """
    NAME:
        grad

    PURPOSE:
        Given an n-dimensional real-valued function, f, return the gradient function of f.

    INPUT:
        f = a differentiable function that takes n number to 1 number
        n = number of variables that f takes

    OUTPUT:
        combined_function = a vector-valued function, with n component functions, each of which takes n numbers to 1
                            number. The component functions are the partial derivative of f with respect to one of its
                            input variable, in the right order

    HISTORY:
        2018-05-23 - Written - Samuel Wong
    """
    # initialize a list of component functions with n components
    component_functions = [None for i in range(n)]

    # loop through each dimension of input of f
    for i in range(n):
        # the ith component function of the gradient is the ith partial derivative of f
        component_functions[i] = partial_derivative(f, i)

    # create the combined gradient fucntion
    def combined_function(x):
        result = np.empty(n)
        for i in range(n):
            result[i] = component_functions[i](x)
        return result

    # return the function we just created. Note that this is just the function, without the argument
    # this means that the result of the return can be called with input
    return combined_function


def evaluate_uniformity(f, x, W):
    """
    NAME:
        evaluate_uniformity

    PURPOSE:
        Given a differentiable function, f, an n-dimensional vector, x, 
        and a m dimensional subscpace of R^n (m <= n) that is spanned by the
        orthonormal vectors contained in W, we first find the gradient of f 
        at x, then normalize the gradient. Next, we take the dot product
        between normalized gradient of f and the vecotrs in w. 
        We return these normalized directional derivatives of f along
        each of the vector. If all of them are close to zero, then f is said to
        be uniform in the subspace.

    INPUT:
        f = a differentiable function that takes a numpy array of n numbers to 1 number
        x = a numpy array containing n component, represents a generic point in R^n
        W = a numpy array that contains m number of n-dimensional linearly independent vectors (warning: requries m<n);
            Each vector is also a numpy array

    OUTPUT:
        directional_derivatives = a numpy array containing the directional derivative of f along each direction
                                  of the basis vectors generating the subspace

    HISTORY:
        2018-05-24 - Written - Samuel Wong
        2018-05-29 - Edited so that it normalize the gradient - Samuel Wong
    """
    # get the size of the subspace; the first coordinate of the shape of W is the number of vectors; the second one
    # is the dimension of each vector
    m = np.shape(W)[0]
    # get the size of the domain spae
    n = np.size(x)
    # get the gradient of f
    del_f = grad(f, n)
    # evaluate the gradient at the given point
    del_f_x = del_f(x)
    # normalize the gradient
    del_f_x = normalize_vector(del_f_x)
    # initialize directional_derivatives, where each component is gradient dotted with one of the vectors
    directional_derivatives = np.empty(m)
    for i in range(m):
        directional_derivatives[i] = np.dot(del_f_x, W[i])
    return directional_derivatives