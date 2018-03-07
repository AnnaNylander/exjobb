import csv
import json
import argparse
import math
import numpy
from util import getEulerDistance, isWithinRadius

parser = argparse.ArgumentParser(description='Create json file')
parser.add_argument('--staticfile', metavar='file.csv', dest='staticfile',
                    default='../../recorded_data/static_measurements/sm.csv',
                    help='file containing static information on speed signs\
                     and traffic lights')
parser.add_argument('--dynamicfile', metavar='file.csv', dest='dynamicfile',
                    default='../../recorded_data/dynamic_measurements/dm.csv',
                    help='file containing dynamic information on traffic lights')
parser.add_argument('--csvfile', metavar='file.csv', dest='csvfile',
                    default='../../recorded_data/player_measurements/pm.csv',
                    help='input file containing postion of car')
parser.add_argument('--jsonfile', metavar='file.json', dest='jsonfile',
                    default='../../recorded_data/traffic_awareness/traffic.json',
                    help='input file containing the path with turns.')

args = parser.parse_args()

RADIUS = 2

def findRelevantTrafficSigns(static_data, car_position, car_yaw):
    result = []
    for row in static_data:
        sign_position = (row['location_y']/100,row['location_x']/100)
        sign_yaw = row['yaw']
        if isWithinRadius(car_position, sign_position, RADIUS) and isVisible(car_yaw, sign_yaw):
            print(car_position)
            result.append(row['id'])
    return result

def isVisible(a, b):
    #-180 till + 180 för båda. De ska vara abs(a-b)> 90
    boolean = math.fabs(a-b) > 90
    print(boolean)
    return boolean


def create_json():
    print("Reading car information from " + args.csvfile +\
            "\nReading static information from " + args.staticfile +\
            "\nReading dynamic information from " + args.dynamicfile +\
            "\nWriting to " + args.jsonfile)

    #read data
    static_data = numpy.genfromtxt(args.staticfile, delimiter=',', names=True)
    dynamic_data = numpy.genfromtxt(args.dynamicfile, delimiter=',', names=True)
    car_data = numpy.genfromtxt(args.csvfile, delimiter=',', names=True)
    #json_data = json.load(open(args.jsonfile))

    data = dict()

    relevant_signs = [] #[id]
    # for every frame in car_data, get relevant traffic signs and TL
    for row in car_data:
        current_position = (row['location_y'],row['location_x'])
        yaw = row['yaw']
        result = findRelevantTrafficSigns(static_data, current_position,yaw)
        relevant_signs = relevant_signs + result

    signs = []
    id = 0
    for sign in relevant_signs:
        if sign == id:
            continue
        signs.append(sign)
        print(sign==id)
        id = sign

    data['values'] = []
    for id in signs:
        index = numpy.where(static_data['id'] == id)
        x_pos = static_data['location_x'][index][0]
        y_pos = static_data['location_y'][index][0]
        type = static_data['type'][index][0]
        type = "light" if type==3 else "sign"
        print(type)
        values = { 'id': id, 'type': type , 'speed_limit': static_data['speed_limit'][index][0]}
        data['values'].append({'location_x': x_pos, 'location_y': y_pos, \
            'data':values})

    last_row = car_data[len(car_data)-1]
    location_y = last_row['location_y']
    location_x = last_row['location_x']
    data['endPosition'] = {'location_x': location_x, 'location_y': location_y, 'data':{'id': 0, 'type':'end', 'speed_limit':0}}
    #traffic sign data: position, id, limit after, limit up until?
    #traffic light data: position, id, status??

    # should I make one for signs and one for TL or one together?
    print(data)


    with open(args.jsonfile,"w") as f:
        json.dump(data,f, sort_keys=False, indent=4, separators=(',', ': '))

if __name__=="__main__":
    create_json()