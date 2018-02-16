import csv
import argparse
from google_intention import getIntentionG
from dead_reckoning import getIntentionDR
from util import calculateDistance, calculateHeading

ALLOWED_TIME_ERROR = 0.001
FUTURE_SECONDS = 5

parser = argparse.ArgumentParser(description='Calculate intentions')
parser.add_argument('--timeInterval', metavar='seconds', type=float,
                    dest='timeInterval', default='1',
                    help='Time interval in seconds between\
                    google intention requests. (float values ok)')
parser.add_argument('--file', metavar='file.csv', dest='file',
                    default='test_data.csv',
                    help='input file')
parser.add_argument('--targetFile', metavar='file.csv', dest='targetFile',
                    default='intentions.csv',
                    help='output file')
parser.add_argument('--latKey', metavar='key', dest='latkey',
                    default='velodyne_gps_latitude',
                    help='csv key to latitude coordinates')
parser.add_argument('--lonKey', metavar='key', dest='lonkey',
                    default='velodyne_gps_longitude',
                    help='csv key to longitude coordinates')
parser.add_argument('--timeKey', metavar='key',  dest='timestamp',
                    default='SentTimeStamp',
                    help='csv key to time stamp')
args = parser.parse_args()

def getDestination(timestamp,seconds):
    """ Find position (in latitude, longitude) X seconds from the current
    timestamp """
    destinationTime = float(timestamp) + seconds;
    csvreader = csv.DictReader(open(args.file), delimiter=';')
    for row in csvreader:
        destination = (float(row[args.latkey]), float(row[args.lonkey]))
        t = float(row[args.timestamp])
        if  abs(t - destinationTime) <= ALLOWED_TIME_ERROR:
            return destination

    #if we get here then timestamp has not been found, the last row is picked.
    return destination

def getIntentions():
    """ Read GPS-IMU values from csv file and calculate intentions.
    Then write the intentions to a new csv file."""
    csvreader = csv.DictReader(open(args.file), delimiter=';')
    print("Reading from " + args.file + " and calculating intentions")
    csvwriter = csv.writer(open(args.targetFile, 'w', newline=''), delimiter=';')
    csvwriter.writerow(['intention_direction', 'intention_proximity'])

    #init values
    intention_type = None
    intention_proximity = 0
    heading = 0.0
    last_position = (0,0)

    for row in csvreader:
        timestamp = float(row[args.timestamp])
        current_position = (float(row[args.latkey]), float(row[args.lonkey]))
        if last_position is None:
            last_position = current_position;

        if (timestamp%args.timeInterval <= ALLOWED_TIME_ERROR):
            destination = getDestination(timestamp,FUTURE_SECONDS)
            (intention_type, intention_proximity) = \
                getIntentionG(current_position, destination)
            heading = calculateHeading(last_position, current_position)
        else:
            (intention_type, intention_proximity) = \
                getIntentionDR(current_position, last_position, heading, \
                                intention_type, intention_proximity)

        csvwriter.writerow([timestamp, intention_type, round(intention_proximity)])
        last_position = current_position

if __name__=="__main__":
    getIntentions()
