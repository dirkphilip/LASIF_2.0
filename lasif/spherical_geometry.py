# Licensed under a 3-clause BSD style license
"""
A collection of classes and functionalities to handle the surfaces
of spherical domains.
When spherical domains are put on a 2D surface grid there is always
a jump in coordinates (i.e. longitude from -180 to 180 in pacific).
This jump is problematic for standard 2D polygon algorithms and
thus needs to be adressed in a specific fashion here.

The implementation is largely inspired, and partially copied from
www.github.com/spacetelescope/spherical_geometry
Thank you for your work on this!
"""
import numpy as np
from lasif.exceptions import LASIFError
from copy import copy, deepcopy
from lasif import great_circle


class SphericalPolygon(object):
    """
    A polygon is represented by (x, y, z) coordinates on a unit sphere and
    an inside point to know which part of the polygon is inside the domain
    and which one is outside the domain.
    """

    def __init__(self, points: np.array, outside: tuple = None):
        """
        Definition of polygon. Contains boundary points and potentially a
        point which is inside the domain

        :param points: Edge coordinates on unit sphere. Nx3 array of x,y,z
        :type points: numpy.ndarray
        :param outside: (x,y,z) triple of an outside coordinate,
            defaults to None
        :type outside: tuple, optional
        """
        if len(points) == 0:
            raise LASIFError("We do not define a polygon of no edge points")
        if not points.shape[1] == 3:
            raise LASIFError("Points should be of Nx3 dimension")

        if not np.array_equal(points[0], points[-1]):
            # We want points to be a closed loop of points
            points = np.concatenate((points, points[0]))

        if points.shape[0] < 4:
            raise LASIFError(
                "A spherical polygon should be defined by at least 3 points"
            )

        self._points = points
        if not self.is_clockwise():
            self._points = points[::-1]

        self._outside = np.asanyarray(outside)

    def __copy__(self):
        return deepcopy(self)

    copy = __copy__

    def __len__(self):
        return 1

    @property
    def points(self):
        """
        The edge points which define the polygon.
        """
        return self._points

    @property
    def outside(self):
        """
        The point which is defined as the inside polygon point
        """
        return self._outside

    def is_clockwise(self):
        """
        Return True if the points in this polygon are in clockwise order.
        The normal vector to the two arcs containing a vertes points outward
        from the sphere if the angle is clockwise and inward if the angle is
        counter-clockwise. The sign of the inner product of the normal vector
        with the vertex tells you this. The polygon is ordered clockwise if
        the vertices are predominantly clockwise and counter-clockwise if
        the reverse.
        """

        points = np.vstack((self._points, self._points[1]))
        A = points[:-2]
        B = points[1:-1]
        C = points[2:]
        orient = great_circle.triple_product(A - B, C - B, B)
        return np.sum(orient) > 0.0

    def contains_point(self, point: tuple) -> bool:
        """
        A method to check whether a point is inside the polygon

        :param point: x,y,z coordinates on a unit sphere. The point to test.
        :type point: tuple
        :return: True if point inside, False if outside
        :rtype: bool
        """
        point = np.asanyarray(point)
        if np.array_equal(self._outside, point):
            return False

        intersects = great_circle.intersects(
            self._points[:-1], self._points[1:], self._outside, point
        )
        crossings = np.sum(intersects)

        return (crossings % 2) != 0
