import csv
import json
import argparse
import math
import numpy
from intentions import getData, getDataTraffic

parser = argparse.ArgumentParser(description='Calculate intentions')
parser.add_argument('--jsonIntentions', metavar='file.json', dest='jsonIntentions',
                    default='../../recorded_data/intentions/intentions.json',
                    help='input file containing the path with turns.')
parser.add_argument('--jsonTraffic', metavar='file.json', dest='jsonTraffic',
                    default='../../recorded_data/traffic_awareness/traffic.json',
                    help='input file containing traffic situation along the path.')
parser.add_argument('--dynamicfile', metavar='file.csv', dest='dynamicfile',
                    default='../../recorded_data/dynamic_measurements/dm.csv',
                    help='file containing dynamic information on traffic lights')
parser.add_argument('--csvfile', metavar='file.csv', dest='csvfile',
                    default='../../recorded_data/player_measurements/pm.csv',
                    help='input file containing information on the car.')
parser.add_argument('--targetIntentions', metavar='file.csv', dest='targetIntentions',
                    default='../../recorded_data/intentions/data.csv',
                    help='output file for intentions')
parser.add_argument('--targetTraffic', metavar='file.csv', dest='targetTraffic',
                    default='../../recorded_data/traffic_awareness/data.csv',
                    help='output file traffic awareness')

args = parser.parse_args()

def main():
    """ Read GPS-IMU values from csv file and calculate intentions.
    Then write the intentions to a new csv file. Also writing traffic intentions."""

    print("****** GENERATING INTENTIONS ******")
    print("Reading car information from " + args.csvfile +\
            "\nreading path from" + args.jsonIntentions)
    car_data = numpy.genfromtxt(args.csvfile, delimiter=',', names=True)
    json_data = json.load(open(args.jsonIntentions))

    print("Preparing to write to: " + args.targetIntentions)
    csvwriter = csv.writer(open(args.targetIntentions, 'w', newline=''), delimiter=',')
    csvwriter.writerow(['frame','intention_proximity','intention_direction'])

    data = getData(car_data, json_data)

    print("Intentions generated. Writing to csv file.")
    for row in data:
        csvwriter.writerow([row[0],round(row[1])] + list(row[2].values()))
    print("Intentions written.")

    print("****** GENERATING TRAFFIC AWARENESS ******")
    print("Reading car information from " + args.csvfile +\
            "\nreading path from " + args.jsonTraffic +\
            "\nreading current traffic status from " + args.dynamicfile)
    car_data = numpy.genfromtxt(args.csvfile, delimiter=',', names=True)
    json_data = json.load(open(args.jsonTraffic))
    dynamic_data = numpy.genfromtxt(args.dynamicfile, delimiter=',', names=True)

    print("Preparing to write to: " + args.targetTraffic)
    csvwriter = csv.writer(open(args.targetTraffic, 'w', newline=''), delimiter=',')
    csvwriter.writerow(['frame','next_distance', 'current_speed_limit','next_speed_limit','light_status'])

    data = getDataTraffic(car_data, json_data, dynamic_data)

    print("Intentions generated. Writing to csv file.")
    for row in data:
        csvwriter.writerow([row[0],round(row[1])] + list(row[2].values()))
    print("Traffic situation written.")


if __name__=="__main__":
    main()
