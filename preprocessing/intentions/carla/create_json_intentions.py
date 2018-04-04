import csv
import json
import argparse
import math
import numpy

#parser = argparse.ArgumentParser(description='Create json file')
#parser.add_argument('--path', metavar='PATH', dest='path',
#                    help='path to recorded_data.')

#args = parser.parse_args()

def create_json(args):
    CSVFILE_PATH = args.data_path + 'player_measurements/pm.csv'
    ANNOTATIONS_PATH = args.data_path + 'intentions/manual_annotations.csv'
    JSON_PATH = args.data_path + 'intentions/intentions.json'

    csv_data = numpy.genfromtxt(CSVFILE_PATH, delimiter=',', names=True)
    annoreader = csv.DictReader(open(ANNOTATIONS_PATH), delimiter=' ')
    print("Reading from " + ANNOTATIONS_PATH +\
            "\n and from " + CSVFILE_PATH +\
            "\n Writing to " + JSON_PATH)

    data = dict()
    values = dict()
    for row in annoreader:
        time = row['time']
        time = time.split(":")
        frame = int (time[0])*60*60*10
        frame = frame + int (time[1])*60*10
        frame = frame + int (time[2])*10
        direction = row['type']
        values[frame] = direction

    frame = 1
    data['values'] = []
    data['startPosition'] = {'location_x': csv_data['location_x'][0], \
                                'location_y': csv_data['location_y'][0]}
    for row in csv_data:
        time = float (row['game_timestamp'])
        if frame in values:
            location_y = row['location_y']
            location_x = row['location_x']
            direction = values[frame]
            dir = 0
            if direction == 'left':
                dir = -1
            if direction =='right':
                dir = 1
            data['values'].append({'location_x': location_x, 'location_y': location_y, \
                'data':{'type': dir}})
        frame = frame+1

    data['endPosition'] = {'location_x': csv_data['location_x'][len(csv_data) -1],\
                            'location_y': csv_data['location_y'][len(csv_data) -1],\
                            'data':{'type':0}}

    with open(JSON_PATH,"w") as f:
        json.dump(data,f, sort_keys=False, indent=4, separators=(',', ': '))

if __name__=="__main__":
    create_json(args)
