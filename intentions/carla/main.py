import csv
import json
import argparse
import math

parser = argparse.ArgumentParser(description='Calculate intentions')
parser.add_argument('--csvfile', metavar='file.csv', dest='csvfile',
                    default='test_data.csv',
                    help='input file containing postion of car')
parser.add_argument('--jsonfile', metavar='file.json', dest='jsonfile',
                    default='path.json',
                    help='input file containing the path with turns.')
parser.add_argument('--targetFile', metavar='file.csv', dest='targetFile',
                    default='intentions.csv',
                    help='output file')
parser.add_argument('--latKey', metavar='key', dest='latkey',
                    default='location_y',
                    help='csv key to latitude coordinates')
parser.add_argument('--lonKey', metavar='key', dest='lonkey',
                    default='location_x',
                    help='csv key to longitude coordinates')

args = parser.parse_args()

def getDistance(startTime, start_pos, end_pos): #TODO every 5 seconds or something instead
    csvreader = csv.DictReader(open(args.csvfile), delimiter=' ')
    interval =  1000 # every one seconds
    sumDistance = 0
    last_position = start_pos
    for row in csvreader:
        t = float (row['game_timestamp'])
        current_position = (float(row[args.latkey]), float(row[args.lonkey]))
        if end_pos == current_position:
            sumDistance = sumDistance + getHypotenuseDistance(last_position, current_position)
            return sumDistance
        if t > startTime and (t-startTime) % interval == 0:
            sumDistance = sumDistance + getHypotenuseDistance(last_position, current_position)
            last_position = current_position



def getHypotenuseDistance(a, b): #TODO every 5 seconds or something instead
    (a_lat,a_lon) = a
    (b_lat,b_lon) = b
    dist = math.sqrt((b_lat - a_lat)**2 + (b_lon - a_lon)**2)
    return dist

def getNextTurn(position):
    (lat,lon) = position
    global turn_index #TODO is this correct syntax?? :s
    if turn_index == max_turn:
        return ((float (data['endPosition']['latitude']),float (data['endPosition']['longitude'])),"end")

    turn = data['turns'][turn_index]
    turn_index = turn_index + 1
    return ((float(turn['latitude']), float(turn['longitude'])),turn['type'])

def getIntentions():
    """ Read GPS-IMU values from csv file and calculate intentions.
    Then write the intentions to a new csv file."""
    csvreader = csv.DictReader(open(args.csvfile), delimiter=' ')
    print("Reading from " + args.csvfile + " and calculating intentions." +\
            " Writing to " + args.targetFile)
    csvwriter = csv.writer(open(args.targetFile, 'w', newline=''), delimiter=';')
    csvwriter.writerow(['frame','intention_direction', 'intention_proximity'])

    #init values
    next_turn_pos = None
    next_turn_type = ""
    frame = 1
    next_turn_distance = 0
    last_position = None
    for row in csvreader:
        current_position = (float(row[args.latkey]), float(row[args.lonkey]))
        time = float (row['game_timestamp'])
        if last_position == None:
            last_position = (float(row[args.latkey]), float(row[args.lonkey]))
        if (next_turn_pos == None or current_position == next_turn_pos):
            next_turn_pos, next_turn_type = getNextTurn(current_position)
            next_turn_distance = getDistance(time, current_position, next_turn_pos)

        # get updated turn proximity
        next_turn_distance = next_turn_distance - getHypotenuseDistance(last_position, current_position)
        next_turn_distance = max(next_turn_distance, 0)
        intention_proximity = next_turn_distance
        intention_type = next_turn_type

        csvwriter.writerow([frame, intention_type, round(intention_proximity)])
        frame = frame+1
        last_position = current_position


if __name__=="__main__":
    global max_turn
    global turn_index
    global data
    turn_index = 0
    data = json.load(open(args.jsonfile))
    max_turn = len(data['turns'])
    getIntentions()
