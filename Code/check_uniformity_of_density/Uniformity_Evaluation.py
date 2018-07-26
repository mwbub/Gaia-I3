"""
NAME:
    Uniformity_Evaluation

PURPOSE:
    This module contains a function that evaluates whether a given function is uniform along a given subspace.
    In addition, it contains functions that return gradient and partial derivative of a given function.

FUNCTIONS:
    evaluate_uniformity: This function takes a differeniable function, a point, and the vectors generating
                         a subspace, and return whether the function is uniform along the subspace at that point.

    grad: This function takes a differentiable function and returns the gradient of that function

    partial_derivative: This function takes a function and the position of the input variable with respect to which
                        the partial derivative is to be taken. Then it returns the partial derivative with respect
                        to that variable.

WARNINGS:
    This module assumes the given function takes a numpy array, which can contain an arbitrary number of values.
    In particular, this code cannot handle a function that takes multiple input directly.

    Also, the derviative function divides given point by a small number: 1e-6
    So when testing this function, do not use integrers. Everything has to be float for this to work.

HISTORY:
    2018-05-24 - Written - Samuel Wong
"""
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


def grad_multi(f, points, dx = 1e-8):
    """
    NAME:
        grad_multi

    PURPOSE:
        Calculate the numerical value of gradient for an array of points, using
        a function that is able to take an array of points

    INPUT:
        f = a differentiable function that takes an array of points, each with n
            dimensions
        points = (m,n) array, representing m points, each with n dimensions

    OUTPUT:
        (m,n) array, each row being a gradient

    HISTORY:
        2018-07-22 - Written - Samuel Wong
    """
    n = np.shape(points)[1]
    increment = dx*np.identity(n)
    df = []
    f_points = f(points)
    for row in increment:
        df.append((f(points + row) - f_points)/dx)
    return np.array(df).T


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


def evaluate_uniformity_dot(f, points, v1, v2):
    """
    NAME:
        evaluate_uniformity_dot

    PURPOSE:
        Given a function <f> that takes (m,6) array of points, and (m,6) array
        of points, <points>, as well as two list of (m,6) vectorrs, <v1> and
        <v2>, return the dot product of gradient f at points witht the 
        orthogonal complement of each pairs of rows in <v1> and <v2>. The four
        dot products being zero implies that f is locally constant along
        corresponding row of <v1> and <v2>.

    INPUT:
        f = a differentiable function that takes an array of points, each with 
            6 dimensions
        points = (m,6) array, representing m points, each with n dimensions
        v1, v2 = an array (m,6) vectors, where each row of v1 and v2 correspond
                to a pair of vectors against which we are testing uniformity

    OUTPUT:
        dot = an (m,4) array, where each row contains 4 dot product
            corresponding to the result of the same row in points, v1, and v2

    HISTORY:
        2018-07-26 - Written - Samuel Wong
    """
    m = np.shape(v1)[0] # get the number of rows
    W = []
    # get the orthogonal complement in each pair of v1 and v2.
    # print if anomaly in number of dimensions
    for i in range(m):
        comp = orthogonal_complement(np.array([v1[i], v2[i]]))
        if np.shape(comp) == (4,6):
            W.append(comp)
        else:
            print("Anomaly at row {}, v1 = {}, v2 = {} , complement = {}"\
                  .format(i, v1[i], v2[i], comp))
    W = np.array(W)
    # reshape so that all 4 orthgoonal vectors for each row are listed together
    # with every 4 adjacent rows being complement of the same gradient
    W = np.reshape(W, (4*m, 6)) 
    # get the normalize gradient and duplicate four times
    # reshape so that every 4 adjacent rows are identical, so that it matches
    # with W to do dot product
    del_f_points = normalize(grad_multi(f, points))
    del_f_points2 = np.copy(del_f_points)
    del_f_points3 = np.copy(del_f_points)
    del_f_points4 = np.copy(del_f_points)
    four_f = np.stack((del_f_points, del_f_points2, del_f_points3,
                       del_f_points4), axis =1)
    four_f = np.reshape(four_f, np.shape(W))
    # get the dot product and reshape such that each row contains 4 dot product
    dot = dot_product(four_f, W)
    dot = np.reshape(dot, (m, 4))
    dot = dot.astype('float64') # change data type from sympy back to float
    return dot

def evaluate_uniformity_projection(points, f, v1, v2):
    """
    NAME:
        evaluate_uniformity_projection

    PURPOSE:
        Calculate the ratio of length of the projection of grad(f)(points)
        on to the space spanned by v1 and v2. A result close to 1 means
        the function is uniform along v1 and v2.

    INPUT:
        f = a differentiable function that takes an array of points, each with n
            dimensions
        points = (m,n) array, representing m points, each with n dimensions

    OUTPUT:
        array of shape (m,), each component represents a fractional length for
        corresponding point

    HISTORY:
        2018-07-22 - Written - Samuel Wong
    """
    p = grad_multi(f, points)
    e1, e2 = Gram_Schmidt_two(v1, v2)
    p_projection = orthogonal_projection(p, e1, e2)
    return LA.norm(p_projection, axis = 1)/LA.norm(p, axis = 1)
    