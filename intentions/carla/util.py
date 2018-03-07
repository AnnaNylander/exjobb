import math

def isWithinRadius(a, b, r):
    if getEulerDistance(a,b) < r:
        return True
    return False

def getEulerDistance(pos_a, pos_b):
    (a_lat,a_lon) = pos_a
    (b_lat,b_lon) = pos_b
    dist = math.sqrt((b_lat - a_lat)**2 + (b_lon - a_lon)**2)
    return dist
