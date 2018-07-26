"""
NAME:
    Uniformity_Evaluation

HISTORY:
    2018-05-24 - Written - Samuel Wong
"""
import numpy as np
from Linear_Algebra import *
from numpy import linalg as LA

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


def evaluate_uniformity(f, points, v1, v2, uniformity_method):
    if uniformity_method == "projection":
        return evaluate_uniformity_projection(f, points, v1, v2)
    elif uniformity_method == "dot product":
        return evaluate_uniformity_dot(f, points, v1, v2)
    else:
        raise Exception("uniformity method not understood.")


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


def evaluate_uniformity_projection(f, points, v1, v2):
    """
    NAME:
        evaluate_uniformity_projection

    PURPOSE:
        Calculate the ratio of length of the projection of grad(f)(points)
        on to the space spanned by v1 and v2. A result close to 1 means
        the function is uniform along v1 and v2.

    INPUT:
        f = a differentiable function that takes an array of points, each with
            n dimensions
            
        points = (m,n) array, representing m points, each with n dimensions
        
        v1, v2 = an array (m,6) vectors, where each row of v1 and v2 correspond
        to a pair of vectors against which we are testing uniformity

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
    