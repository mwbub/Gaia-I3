"""
NAME:
    Linear_Algebra

PURPOSE:
    This module contains some linear algebra tools, most notably a function that computes the orthogonal complement
    space.

FUNCTIONS:
    orthogonal_complement: this function returns the orthogonal complement space of a given set of vectors

HISTORY:
    2018-05-25 - Written - Samuel Wong
"""
import numpy as np
from sympy import Matrix, GramSchmidt


def orthogonal_complement(V):
    """
    NAME:
        orthogonal_complement

    PURPOSE:
        Given V, a set of m number of n-dimensional vectors, find the orthogonal complement to the space span by
        the m vectors. Then return an orthonormal basis of the orthogonal space.

    INPUT:
        V = a numpy array containing m number of n-dimensional vectors; the vectors are represented in rows

    OUTPUT:
        W = a numpy array containing a number of linearly independent n-dimensional vectors that form an
            orthonormal basis of the orthogonal complement of span(V)

    WARNING: the shape of V must be in the form of (m,n), with vectors written in row. Also, the function will return
             an empty array if V already has maximum rank

    HISTORY:
        2018-05-25 - Written - Samuel Wong
    """
    # the actual code is simple but the math behind is a bit tricky. I will explain the behind the scene math here.
    # take the transpose of V so that the vectors spanning the space are written in column: A = V.T
    # A is an n by m matrix
    # R(A) is the column space of A, or the range of A, which is also the space to be eliminated from R^n to get the
    # orthogonal complement
    # so we want W = R(A)^{perp}, the orthogonal complement of range of A
    # But by a basic theorem, R(A)^{perp} = N(A.T) = N(V)
    # so we just need to find the null space of V

    # convert V to a sympy Matrix object
    V = Matrix(V)
    # get the null space of V. The vectors of the null space are stored as rows in W.
    W = V.nullspace()
    # apply the GramScmidt process to orthogonalize the vectors in W. Adding the True statement also normalize them
    W = GramSchmidt(W, True)
    # convert W back into numpy array
    W = np.array(W)
    return W

def normalize_vector(v):
    """
    NAME:
        normalize_vector

    PURPOSE:
        Given a vector v, return the normalized versioon.

    INPUT:
        v = a numpy array containing n components

    OUTPUT:
        a vector in the same directin as v but has unit length

    HISTORY:
        2018-05-30 - Written - Samuel Wong
    """
    v = Matrix(v)
    v = v.normalized()
    v = np.array(v.T)
    return v[0]

def dot_product(a, b):
    """
    NAME:
        dot_product

    PURPOSE:
        Given two 2D numpy array, <a> and <b>, where each row represent a
        vector and they both are lists of the same number of vectors, return 
        a list of dot product of corresponding rows.

    INPUT:
        a, b = (m by n) numpy array representing m vectors, each of n
                dimensions

    OUTPUT:
        1D numpy array of shape (n,), with each component being a dot product

    HISTORY:
        2018-07-22 - Written - Samuel Wong
    """
    # multiply two matrix compoenet-wise
    product = np.multiply(a,b)
    # add up the columns
    return np.sum(product, axis = 1)