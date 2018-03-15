import csv
import json
import argparse
import math
import numpy
from intentions.carla.get_intentions import getData, getDataTraffic
from intentions.carla.create_json_intentions import create_json as intentions_json
from intentions.carla.create_json_traffic import create_json as traffic_json


#parser = argparse.ArgumentParser(description='Calculate intentions')
#parser.add_argument('--path', metavar='PATH', dest='data_path',
#                    help='path to recorded_data (with trailing / ).')

#args = parser.parse_args()

def main(args):
    """ Read GPS-IMU values from csv file and calculate intentions.
    Then write the intentions to a new csv file. Also writing traffic intentions."""

    # create paths.
    DYNAMIC_PATH = args.data_path + 'dynamic_measurements/dm.csv'
    CSVFILE_PATH = args.data_path + 'player_measurements/pm.csv'
    JSON_INTENTIONS = args.data_path + 'intentions/intentions.json'
    JSON_TRAFFIC = args.data_path + 'traffic_awareness/traffic.json'
    TARGET_TRAFFIC_PATH = args.data_path + 'traffic_awareness/traffic.csv'
    TARGET_INTENTIONS_PATH = args.data_path + 'intentions/intentions.csv'


    print("****** Creating json files ******")
    print("\t json for intentions")
    intentions_json(args)
    print("\t json for traffic situation")
    traffic_json(args)
    print("Json generated.")

    print("****** GENERATING INTENTIONS ******")
    print("Reading car information from " + CSVFILE_PATH +\
            "\nreading path from" + JSON_INTENTIONS)
    car_data = numpy.genfromtxt(CSVFILE_PATH, delimiter=',', names=True)
    json_data = json.load(open(JSON_INTENTIONS))

    print("Preparing to write to: " + TARGET_INTENTIONS_PATH)
    csvwriter = csv.writer(open(TARGET_INTENTIONS_PATH, 'w', newline=''), delimiter=',')
    csvwriter.writerow(['frame','intention_proximity','intention_direction'])

    data = getData(car_data, json_data)

    print("\tIntentions generated. Writing to csv file.")
    for row in data:
        csvwriter.writerow([row[0],round(row[1])] + list(row[2].values()))
    print("\tIntentions written.")

    print("****** GENERATING TRAFFIC AWARENESS ******")
    print("Reading car information from " + CSVFILE_PATH +\
            "\nreading path from " + JSON_TRAFFIC +\
            "\nreading current traffic status from " + DYNAMIC_PATH)
    car_data = numpy.genfromtxt(CSVFILE_PATH, delimiter=',', names=True)
    json_data = json.load(open(JSON_TRAFFIC))
    dynamic_data = numpy.genfromtxt(DYNAMIC_PATH, delimiter=',', names=True)

    print("Preparing to write to: " + TARGET_TRAFFIC_PATH)
    csvwriter = csv.writer(open(TARGET_TRAFFIC_PATH, 'w', newline=''), delimiter=',')
    csvwriter.writerow(['frame','next_distance', 'current_speed_limit','next_speed_limit','light_status'])

    data = getDataTraffic(car_data, json_data, dynamic_data)

    print("\tIntentions generated. Writing to csv file.")
    for row in data:
        csvwriter.writerow([row[0],round(row[1])] + list(row[2].values()))
    print("\tTraffic situation written.")


if __name__=="__main__":
    main(args)
