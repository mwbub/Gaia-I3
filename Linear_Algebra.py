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
