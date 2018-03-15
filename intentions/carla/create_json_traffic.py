import csv
import json
import argparse
import math
import numpy
from intentions.carla.util import getEulerDistance, isWithinRadius

#parser = argparse.ArgumentParser(description='Create json file')
#parser.add_argument('--path', metavar='PATH', dest='path',
#                    help='path to recorded_data.')

#args = parser.parse_args()

RADIUS = 4

def findRelevantTrafficSigns(static_data, car_position, car_yaw):
    result = []
    for row in static_data:
        sign_position = (row['location_y'],row['location_x'])
        sign_yaw = row['yaw']
        if isWithinRadius(car_position, sign_position, RADIUS) and isVisible(car_yaw, sign_yaw):
            result.append(row['id'])
    return result

def isVisible(a, b):
    #-180 till + 180 för båda. De ska vara abs(a-b)> 90
    boolean = math.fabs(a-b) > 90
    return boolean


def create_json(args):
    DYNAMIC_PATH = args.data_path + 'dynamic_measurements/dm.csv'
    STATIC_PATH = args.data_path + 'static_measurements/sm.csv'
    CSVFILE_PATH = args.data_path + 'player_measurements/pm.csv'
    JSON_PATH = args.data_path + 'traffic_awareness/traffic.json'

    print("Reading car information from " + CSVFILE_PATH +\
            "\nReading static information from " + STATIC_PATH +\
            "\nReading dynamic information from " + DYNAMIC_PATH +\
            "\nWriting to " + JSON_PATH)

    #read data
    static_data = numpy.genfromtxt(STATIC_PATH, delimiter=',', names=True)
    dynamic_data = numpy.genfromtxt(DYNAMIC_PATH, delimiter=',', names=True)
    car_data = numpy.genfromtxt(CSVFILE_PATH, delimiter=',', names=True)
    #json_data = json.load(open(JSON_PATH))

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
        id = sign

    data['values'] = []
    for id in signs:
        index = numpy.where(static_data['id'] == id)
        x_pos = static_data['location_x'][index][0]
        y_pos = static_data['location_y'][index][0]
        type = static_data['type'][index][0]
        values = { 'id': id, 'type': type , 'speed_limit': static_data['speed_limit'][index][0]}
        data['values'].append({'location_x': x_pos, 'location_y': y_pos, \
            'data':values})

    last_row = car_data[len(car_data)-1]
    location_y = last_row['location_y']
    location_x = last_row['location_x']
    data['endPosition'] = {'location_x': location_x, 'location_y': location_y, 'data':{'id': 0, 'type':0, 'speed_limit':0}}


    with open(JSON_PATH,"w") as f:
        json.dump(data,f, sort_keys=False, indent=4, separators=(',', ': '))

if __name__=="__main__":
    create_json(args)
