import math
from util import calculateDistance, calculateHeading

def getIntentionDR(new_pos, last_pos, last_heading, intention_type, intention_proximity):
    """ Get an intention by calculating the difference in position and
    heading."""
    distance =calculateDistance(last_pos, new_pos);

    heading = calculateHeading(last_pos, new_pos);

    diff_heading = math.fabs(last_heading - heading)

    # If we drive in the opposite direction as before (i.e. backing), then we
    # add to the intention proximity, otherwise we assume progress on the road.
    # (No unexpected turns can happen since this is only for supervised training,
    # and we know for certain our future position in x seconds)
    if diff_heading >= 150:
        return (intention_type, intention_proximity + distance)
    else:
        return (intention_type, intention_proximity - distance)
