import googlemaps
import json
import re
#import LatLon #latlon är typ trasigt. no kidding.
from datetime import datetime

gmaps = googlemaps.Client(key='AIzaSyBTWamksTAzYtq_oUaz8qaR9rLGYPGWWcc')
# All locations will be in latitude,longitude format.

def parseResult(directions_result):
    turn_coord = None
    turn_type = None
    distance_to_turn = 0;

    data = json.loads(json.dumps(directions_result))

    steps = data[0]['legs'][0]['steps'] # is 'steps' too specific?

    for step in steps:
        instruction = json.dumps(step['html_instructions'])
        regex = re.search('right|left|u-turn|take exit', instruction, flags=re.IGNORECASE)
        if regex is not None:
        # if we find a turn, get it's location and return values.
            turn_type = regex.group(0)
            turn_coord = (step['start_location']['lat'],step['start_location']['lng'])
            return turn_type, turn_coord, distance_to_turn
        else:
            distance = json.dumps(step['distance']['value'])
            distance_to_turn += int(distance)

    # no turn found
    return turn_type, turn_coord, -1

#validate distance
def calculateProximityDistanceMatrix(origin,location):
    # calculate with google distance matrix
    distance_matrix = gmaps.distance_matrix(origin, location, mode="driving")
    distance  = json.dumps(distance_matrix['rows'][0]['elements'][0]['distance']['value'])

    return int(distance)

def getDestination():
    destination = (57.6869636,11.9828744) #dummy value
    return destination

def getOrigin():
    origin = (57.6873622, 11.9820609) #dummy value
    return origin

def getIntentions():

    origin = getOrigin()
    destination = getDestination()

    directions_result = gmaps.directions(origin, destination, mode="driving")
    (turn_type, turn_coord, distance_to_turn) = parseResult(directions_result)
    print(turn_type, turn_coord, distance_to_turn)

    # calculateProximityDistanceMatrix is used as validation atm.
    # If we are to keep it as validation, refractor into "validateProximity" function
    proximity = calculateProximityDistanceMatrix(origin,turn_coord)
    if distance_to_turn != proximity:
        print("[UH-OH!!] Proximity validation failed: --------------- \n" \
            + "Directions think: " + str(distance_to_turn) + " m. " \
            "Distance matrix think: " + str(proximity) + " m.")


if __name__=="__main__":
    getIntentions()
