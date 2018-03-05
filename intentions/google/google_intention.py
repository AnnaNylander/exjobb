import googlemaps
import json
import re
from util import calculateHeading, calculateDistance

f = open('google_api_key.txt', 'r')
google_api_key=f.read().replace('\n', '')
gmaps = googlemaps.Client(key=google_api_key)

def parseResult(directions_result):
    """ Parse result form google maps API's directions-function. """
    turn_coord = None
    turn_type = None
    distance_to_turn = 0;

    data = json.loads(json.dumps(directions_result))

    steps = data[0]['legs'][0]['steps']
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
            distance_to_turn += float(distance)

    # no turn found
    leg = data[0]['legs'][0]
    return "forward", (leg['end_location']['lat'],leg['end_location']['lng']), leg['distance']['value']

def getIntentionG(origin, destination):
    """ Get intentions from Google Maps API. """
    directions_result = gmaps.directions(origin, destination, mode="driving")
    (turn_type, turn_coord, distance_to_turn) = parseResult(directions_result)
    return [turn_type, distance_to_turn]
