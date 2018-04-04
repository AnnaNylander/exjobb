from geopy.distance import great_circle
import math

def calculateDistance(origin, destination):
    """ Calculates the distance between two coordinates. It is offline and
    restricted by the assumption that earth is a perfect sphere. """

    return float((great_circle(origin, destination).km)*1000)

def calculateHeading(origin, destination):
    """ Calculate the heading direction between two coordinates. It returns the
    heading in degrees, where 0 deg is North. (This is not very accurate but
    good enough for us.)"""
    x1 = destination[0]
    y1 = destination[1]
    x2 = origin[0]
    y2 = origin[1]

    degrees = math.degrees(math.atan2((y1 - y2), (x1 - x2)))
    degrees = degrees + 90 # North is 0 deg
    return degrees
