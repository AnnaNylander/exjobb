import csv
import json
import argparse
import math

parser = argparse.ArgumentParser(description='Create json file')
parser.add_argument('--annofile', metavar='file.csv', dest='annotations',
                    default='../../recorded_data/intentions/manual_annotations.csv',
                    help='input file containing postion of car')
parser.add_argument('--csvfile', metavar='file.csv', dest='csvfile',
                    default='../../recorded_data/player_measurements/pm.csv',
                    help='input file containing postion of car')
parser.add_argument('--jsonfile', metavar='file.json', dest='jsonfile',
                    default='../../recorded_data/intentions/intentions.json',
                    help='input file containing the path with turns.')

args = parser.parse_args()

def create_json():
    csvreader = csv.DictReader(open(args.csvfile), delimiter=',')
    annoreader = csv.DictReader(open(args.annotations), delimiter=' ')
    print("Reading from " + args.annotations +\
            "\n and from " + args.csvfile +\
            "\n Writing to " + args.jsonfile)

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

    frame = 1 #offset because data starts at row 1
    last_row = {}
    data['values'] = []
    for row in csvreader:
        time = float (row['game_timestamp'])
        if time == 100: #TODO denna är väldigt oflexibel
            location_y = row['location_y']
            location_x = row['location_x']
            data['startPosition'] = {'location_x': location_x, 'location_y': location_y}
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
        last_row = row
        frame = frame+1

    location_y = last_row['location_y']
    location_x = last_row['location_x']
    data['endPosition'] = {'location_x': location_x, 'location_y': location_y, 'data':{'type':0}}

    with open(args.jsonfile,"w") as f:
        json.dump(data,f, sort_keys=False, indent=4, separators=(',', ': '))

if __name__=="__main__":
    create_json()
