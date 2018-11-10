import copy
import math
import numpy as np
import gpxpy


def align_two_tracks(track1, track2, gap_penalty):
    """
    Needleman-Wunsch algorithm adapted for gps tracks.
    """
    # print("Aligning tracks")

    def similarity(p1, p2):
        d = gpxpy.geo.distance(p1.latitude, p1.longitude, p1.elevation,
                               p2.latitude, p2.longitude, p2.elevation)
        return -d

    # construct f-matrix
    f = np.zeros((len(track1), len(track2)))
    for i in range(0, len(track1)):
        f[i][0] = gap_penalty * i
    for j in range(0, len(track2)):
        f[0][j] = gap_penalty * j
    for i in range(1, len(track1)):
        t1 = track1[i]
        for j in range(1, len(track2)):
            t2 = track2[j]
            match = f[i - 1][j - 1] + similarity(t1, t2)
            delete = f[i - 1][j] + gap_penalty
            insert = f[i][j - 1] + gap_penalty
            f[i, j] = max(match, max(delete, insert))
    # backtrack to create alignment
    a1 = []
    a2 = []
    i = len(track1) - 1
    j = len(track2) - 1
    while i > 0 or j > 0:
        if i > 0 and j > 0 and \
           f[i, j] == f[i - 1][j - 1] + similarity(track1[i], track2[j]):
            a1.insert(0, track1[i])
            a2.insert(0, track2[j])
            i -= 1
            j -= 1
        elif i > 0 and f[i][j] == f[i - 1][j] + gap_penalty:
            a1.insert(0, track1[i])
            a2.insert(0, None)
            i -= 1
        elif j > 0 and f[i][j] == f[i][j - 1] + gap_penalty:
            a1.insert(0, None)
            a2.insert(0, track2[j])
            j -= 1
    return a1, a2


def interpolate_distance(points, distance):
    """
    Interpolates points so that the distance between each point is equal
    to `distance` in meters.

    Only latitude and longitude are interpolated; time and elavation are not
    interpolated and should not be relied upon.
    """
    # TODO: Interpolate elevation and time.
    # print("Distributing points evenly every {} meters".format(distance))

    d = 0
    i = 0
    even_points = []
    while i < len(points):
        if i == 0:
            even_points.append(points[0])
            i += 1
            continue

        if d == 0:
            p1 = even_points[-1]
        else:
            p1 = points[i - 1]
        p2 = points[i]

        d += gpxpy.geo.distance(p1.latitude, p1.longitude, p1.elevation,
                                p2.latitude, p2.longitude, p2.elevation)

        if d >= distance:
            brng = bearing(p1, p2)
            ld = gpxpy.geo.LocationDelta(distance=-(d - distance), angle=brng)
            p2_copy = copy.deepcopy(p2)
            p2_copy.move(ld)
            even_points.append(p2_copy)

            d = 0
        else:
            i += 1
    else:
        even_points.append(points[-1])

    return even_points


def bearing(point1, point2):
    """
    Calculates the initial bearing between point1 and point2 relative to north
    (zero degrees).
    """

    lat1r = math.radians(point1.latitude)
    lat2r = math.radians(point2.latitude)
    dlon = math.radians(point2.longitude - point1.longitude)

    y = math.sin(dlon) * math.cos(lat2r)
    x = math.cos(lat1r) * math.sin(lat2r) - math.sin(lat1r) \
                        * math.cos(lat2r) * math.cos(dlon)
    return math.degrees(math.atan2(y, x))
