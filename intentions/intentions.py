import googlemaps
import json
import re
import csv
import argparse
#import LatLon #latlon är typ trasigt. no kidding.
from datetime import datetime
from geopy.distance import great_circle # this accuracy is enough.

f = open('google_api_key.txt', 'r')
google_api_key=f.read().replace('\n', '')
gmaps = googlemaps.Client(key=google_api_key)

parser = argparse.ArgumentParser(description='Calculate intentions')
parser.add_argument('--file', metavar='file.csv', dest='file',
                    default='test_data.csv',
                    help='the file to read from')
parser.add_argument('--targetFile', metavar='file.csv', dest='targetFile',
                    default='intentions.csv',
                    help='the file to write to')
parser.add_argument('--latkey', metavar='key', dest='latkey',
                    default='velodyne_gps_latitude',
                    help='csv key to latitude coordinates')
parser.add_argument('--lonkey', metavar='key', dest='lonkey',
                    default='velodyne_gps_longitude',
                    help='csv key to longitude coordinates')
parser.add_argument('--timekey', metavar='key',  dest='timestamp',
                    default='SentTimeStamp',
                    help='csv key to time stamp')
args = parser.parse_args()

TIME_ERROR = 0.01
FUTURE_SECONDS = 5
# All locations will be in latitude,longitude format.

def parseResult(directions_result):
    turn_coord = None
    turn_type = None
    distance_to_turn = 0;

    data = json.loads(json.dumps(directions_result))

    #TODO if no turn???
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

def calcualteProximityGeopy(origin, destination):
    distance =(great_circle(origin, destination).km)*1000

    return int(round(distance))

def validateDistance(origin,destination,distance_to_turn):
    proximityDM = calculateProximityDistanceMatrix(origin,destination)

    proximityGeopy = calcualteProximityGeopy(origin, destination)

    if ((distance_to_turn != proximityDM) or (distance_to_turn != proximityGeopy)):
        print("[UH-OH!!] Proximity validation failed: --------------- \n" \
            + "Directions think: " + str(distance_to_turn) + " m. \n" \
            + "Distance matrix think: " + str(proximityDM) + " m. \n" \
            + "Geopy think: " + str(proximityGeopy) + " m. ")
    return

def getDestination(time):
    destinationTime = float(time) + FUTURE_SECONDS;
    csvreader = csv.DictReader(open("test_data.csv"), delimiter=';')
    for row in csvreader:
        destination = (float(row['velodyne_gps_latitude']),float(row['velodyne_gps_longitude']))
        t = float(row['SentTimeStamp'])
        if  abs(t - destinationTime) <= TIME_ERROR: # TODO destinationTime not always exact.
            return destination

    #if we get here then timestamp has not been found, pick the last row.
    return destination

def getIntention(origin, destination):
    directions_result = gmaps.directions(origin, destination, mode="driving")
    (turn_type, turn_coord, distance_to_turn) = parseResult(directions_result)
    #validate proximity
    validateDistance(origin,turn_coord,distance_to_turn)
    return [turn_type, distance_to_turn]

def getIntentions():
    list_of_intentions = list()
    csvreader = csv.DictReader(open(args.file), delimiter=';')

    #try:
    print("Reading from " + args.file + " and calculating intentions")
    for row in csvreader:
        timestamp = row[args.timestamp]
        origin = (float(row[args.latkey]), float(row[args.lonkey]))
        destination = getDestination(timestamp)
        #dummy values because data is inconsistent.
        origin = (57.70900, 11.9820609)
        destination = (57.6969636,11.9828744)
        intention = getIntention(origin, destination)
        list_of_intentions.append([timestamp] + intention)
    #except:
    #    print()

    print("writing intentions to csv file: " + args.targetFile)
    csvwriter = csv.writer(open(args.targetFile, 'w', newline=''), delimiter=';')
    csvwriter.writerow(['intention_direction', 'intention_proximity'])
    csvwriter.writerows(list_of_intentions)




if __name__=="__main__":
    getIntentions()
