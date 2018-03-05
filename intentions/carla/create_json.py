import csv
import json
import argparse
import math

parser = argparse.ArgumentParser(description='Create json file')
parser.add_argument('--annofile', metavar='file.csv', dest='annotations',
                    default='annotations.csv',
                    help='input file containing postion of car')
parser.add_argument('--csvfile', metavar='file.csv', dest='csvfile',
                    default='test_data.csv',
                    help='input file containing postion of car')
parser.add_argument('--jsonfile', metavar='file.json', dest='jsonfile',
                    default='path.json',
                    help='input file containing the path with turns.')

args = parser.parse_args()

def create_json():
    csvreader = csv.DictReader(open(args.csvfile), delimiter=' ')
    annoreader = csv.DictReader(open(args.annotations), delimiter=' ')
    print("Reading from " + args.annotations +\
            "and from " + args.csvfile +\
            " Writing to " + args.jsonfile)

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
    data['turns'] = []
    for row in csvreader:
        time = float (row['game_timestamp'])
        if time == 100: #TODO denna är väldigt oflexibel
            latitude = row['location_y']
            longitude = row['location_x']
            data['startPosition'] = {'longitude': longitude, 'latitude': latitude}
        if frame in values:
            latitude = row['location_y']
            longitude = row['location_x']
            data['turns'].append({'longitude': longitude, 'latitude': latitude, 'type': values[frame]})
        last_row = row
        frame = frame+1

    latitude = last_row['location_y']
    longitude = last_row['location_x']
    data['endPosition'] = {'longitude': longitude, 'latitude': latitude}

    with open(args.jsonfile,"w") as f:
        json.dump(data,f, sort_keys=True, indent=4, separators=(',', ': '))

if __name__=="__main__":
    create_json()
