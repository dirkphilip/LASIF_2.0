# Licensed under a 3-clause BSD style license
"""
Utility functions related to computing great circle paths.
Great circles are paths on a sphere with the center of the circle
in the middle of the sphere.
Here we use the unit circle.

Based on and partially copied from:
www.github.com/spacetelescope/spherical_geometry
"""
import numpy as np
from numpy.core.umath_tests import inner1d


def triple_product(A: np.ndarray, B: np.ndarray, C: np.ndarray):
    return inner1d(C, np.cross(A, B))


def _cross_and_normalize(A: np.ndarray, B: np.ndarray):
    T = np.cross(A, B)
    # Normalization
    l = np.sqrt(np.sum(T ** 2, axis=-1))
    l = two_d(l)
    # Might get some divide-by-zeros
    with np.errstate(invalid="ignore"):
        TN = T / l
    # ... but set to zero, or we miss real NaNs elsewhere
    TN = np.nan_to_num(TN)
    return TN


def two_d(vec: np.ndarray):
    """
    Reshape a one dimensional vector so it has a second dimension
    """
    shape = list(vec.shape)
    shape.append(1)
    shape = tuple(shape)
    return np.reshape(vec, shape)


def intersection(A: np.ndarray, B: np.ndarray, C: tuple, D: tuple):
    """
    Returns the point(s) of intersection between two great circle arcs.
    The arcs are defined between the points *AB* and *CD*.  Either *A*
    and *B* or *C* and *D* may be arrays of points, but not both.

    :param A: An array of Nx3 dimension including x,y,z tuples describing
        a path on a sphere
    :type A: numpy.ndarray
    :param B: Same as A
    :type B: numpy.ndarray
    :param C: A tuple with (x, y, z) coordinates on a unit sphere
    :type C: tuple
    :param D: Same as C
    :type D: tuple
    """

    A = np.asanyarray(A)
    B = np.asanyarray(B)
    C = np.asanyarray(C)
    D = np.asanyarray(D)

    A, B = np.broadcast_arrays(A, B)
    C, D = np.broadcast_arrays(C, D)

    ABX = np.cross(A, B)
    CDX = np.cross(C, D)
    T = _cross_and_normalize(ABX, CDX)
    T_ndim = len(T.shape)

    if T_ndim > 1:
        s = np.zeros(T.shape[0])
    else:
        s = np.zeros(1)
    s += np.sign(inner1d(np.cross(ABX, A), T))
    s += np.sign(inner1d(np.cross(B, ABX), T))
    s += np.sign(inner1d(np.cross(CDX, C), T))
    s += np.sign(inner1d(np.cross(D, CDX), T))
    if T_ndim > 1:
        s = two_d(s)

    cross = np.where(s == -4, -T, np.where(s == 4, T, np.nan))

    # If they share a common point, it's not an intersection.  This
    # gets around some rounding-error/numerical problems with the
    # above.
    equals = (
        np.all(A == C, axis=-1)
        | np.all(A == D, axis=-1)
        | np.all(B == C, axis=-1)
        | np.all(B == D, axis=-1)
    )

    equals = two_d(equals)

    return np.where(equals, np.nan, cross)


def intersects(A: np.ndarray, B: np.ndarray, C: tuple, D: tuple):
    """
    Returns `True` if the great circle arcs between *AB* and *CD*
    intersect.  Either *A* and *B* or *C* and *D* may be arrays of
    points, but not both.

    :param A: An array of Nx3 dimension including x,y,z tuples describing
        a path on a sphere
    :type A: numpy.ndarray
    :param B: Same as A
    :type B: numpy.ndarray
    :param C: A tuple with (x, y, z) coordinates on a unit sphere
    :type C: tuple
    :param D: Same as C
    :type D: tuple
    """
    with np.errstate(invalid="ignore"):
        intersections = intersection(A, B, C, D)

    return np.isfinite(intersections[..., 0])
