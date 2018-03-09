"""Base distance measures"""

import math

from scipy.spatial import distance

def hausdorff_origin(interval1, interval2):
    """Hausdorff distance to the origin.
    Assuming interval2 is the origin [0,1)

    Attributes:
        interval1 -- Array of the form Nx2
    """
    return interval1[:, 0] + interval1[:, 1]


def hausdorff_dist(interval1, interval2):
    """Hausdorff distance between interval1 and interval2.

    Attributes:
        interval1 -- Array of the form Nx2 [[start, length], ...]
        interval2 -- See interval1
    """
    interval1 = interval1.reshape(-1, 1)
    interval2 = interval2.reshape(-1, 1)
    dist1 = distance.directed_hausdorff(interval1, interval2)
    dist2 = distance.directed_hausdorff(interval2, interval1)

    return max(dist1[0], dist2[0])


def circle_dist(origin, destination):
    """Great circle dist between the two given points.
    Coordinates are assumed to be longitude and latitude

    Attributes:
        origin -- (lon, lat)
        destination -- (lon, lat)
    """

    lon1, lat1 = origin
    lon2, lat2 = destination
    radius = 6371000
    phi_1 = math.radians(lat1)
    phi_2 = math.radians(lat2)

    delta_phi = math.radians(lat2-lat1)
    delta_lambda = math.radians(lon2-lon1)

    a = math.sin(delta_phi/2.0)**2 +\
        math.cos(phi_1)*math.cos(phi_2) *\
        math.sin(delta_lambda/2.0)**2
    c = 2*math.asin(math.sqrt(a))

    meters = radius*c

    return meters/1000.0
