import csv
import json
import argparse
import math
from util import getEulerDistance, isWithinRadius

RADIUS = 4

def getTrueDistance(car_data, start_pos, end_pos):
    sumDistance = 0
    last_position = start_pos
    for row in car_data:
        current_position = (row['location_y'],row['location_x'])
        sumDistance = sumDistance + getEulerDistance(last_position, current_position)
        if isWithinRadius(end_pos,current_position,RADIUS):
            return sumDistance
        last_position = current_position

def getNextJson(index,data, position):
    (lat,lon) = position
    global max_index
    max_index = len(data['values'])
    if index >= max_index:
        return ((float (data['endPosition']['location_y']),\
            float (data['endPosition']['location_x'])),\
            data['endPosition']['data'])

    values = data['values'][index]
    return ((float(values['location_y']), float(values['location_x'])),values['data'])

def getData(car_data, json_data):
    """used for turn intentions"""
    next_pos = None
    next_value = ""
    frame = 0
    next_distance = 0

    last_position = None
    data = []
    json_index = 0

    for row in car_data:
        current_position = (row['location_y'],row['location_x'])
        time = float (row['game_timestamp'])
        if last_position == None:
            last_position = current_position
        if (next_pos == None or isWithinRadius(current_position,next_pos,RADIUS)):
            next_pos, next_value = getNextJson(json_index,json_data, current_position)
            json_index = json_index+1
            next_distance = getTrueDistance(car_data[frame:len(car_data)], current_position, next_pos)

        # get updated turn proximity
        next_distance = next_distance - getEulerDistance(last_position, current_position)
        next_distance = max(next_distance, 0)

        data.append([frame, next_distance, next_value])
        frame = frame+1
        last_position = current_position

    return data


def getDataTraffic(car_data, json_data, dynamic_data):
    """use for traffic"""
    next_pos = None
    next_value = None
    frame = 0
    next_distance = 0

    last_position = None
    data = []
    json_index = 0
    isTrafficLight = False
    isEnd = False
    current_speed_limit = 0;
    next_speed_limit = 30;

    for row in car_data:
        current_position = (row['location_y'],row['location_x'])
        time = float (row['game_timestamp'])
        if last_position == None:
            last_position = current_position
        if (next_pos == None or isWithinRadius(current_position,next_pos,RADIUS)):
            if next_speed_limit is not None:
                current_speed_limit = next_speed_limit
            next_pos, next_value = getNextJson(json_index,json_data, current_position)
            json_index = json_index+1
            next_distance = getTrueDistance(car_data[frame:len(car_data)], current_position, next_pos)
            if next_value['type'] == 3:
                isTrafficLight = True
            else:
                isTrafficLight = False
            if next_value['type'] == 0:
                isEnd = True

        # get updated turn proximity
        next_distance = next_distance - getEulerDistance(last_position, current_position)
        next_distance = max(next_distance, 0)

        light_status = None
        next_speed_limit = None
        if isTrafficLight:
            id = str(int(next_value['id']))
            light_status = dynamic_data[id][frame]
        elif not isEnd:
            next_speed_limit = next_value['speed_limit']
        values = {'current_speed_limit':current_speed_limit,'next_speed_limit':next_speed_limit,'light_status':light_status}
        if isEnd:
            values = {'current_speed_limit':current_speed_limit,'next_speed_limit':None,'light_status':None}
        data.append([frame, next_distance, values])
        frame = frame+1
        last_position = current_position

    return data
