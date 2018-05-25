"""
NAME:
    Linear_Algebra

PURPOSE:
    This module contains some linear algebra tools, most notably a function that computes the orthogonal complement
    space.

FUNCTIONS:
    orthonormal: the function takes a collection of linearly independent vectors and return an orthonormal basis
    orthogonal_complement: this function returns the orthogonal complement space of a given set of vectors

HISTORY:
    2018-05-24 - Written - Samuel Wong
"""
import numpy as np
from sympy import *


def orthogonal_complement(n, V):
    """
    NAME:
        dimensional_elimination

    PURPOSE:
        Given the general n dimensional real number space, R^n, and V, a set of m vectors (m < n) in the space,
        find the orthogonal complement to the space span by the m vectors. Then return an orthonormal basis of the
        dimensionally reduced space.

    INPUT:
        n = the number of dimensions of the space. The vector space is R^n.
        V = a numpy array containing m number of linearly independent n-dimensional vectors, where m < n

    OUTPUT:
        W = a numpy array containing (m-n) number of linearly independent n-dimensional vectors that form an
            orthonormal basis of the orthogonal complement of span(V)

    HISTORY:
        2018-05-24 - Written - Samuel Wong
    """
    # get the number of vectors in V
    m = np.size(V)
    # take the transpose of V so that the vectors spanning the space are written in column
    A = V.T  # A is now an m by n matrix
    # R(A) is the column space of A, or the range of A, which is also the space to be eliminated from R^n to get the
    # orthogonal complement
    # so we want W = R(A)^{perp}. But by a basic theorem, R(A)^{perp} = N(A.T) = N(V)
    # so we just need null space of V
    W = null_space(V)
    W = orthonormal(W)
    return W
