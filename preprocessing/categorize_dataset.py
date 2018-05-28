import os
import argparse
import shutil
import pandas as pd
import numpy as np

''' Example:

python3 categorize_dataset.py -a train/apple/categories/apple.csv -s train/apple/categories/categories.csv
'''

parser = argparse.ArgumentParser(description='Split data into categories')

parser.add_argument('-a','--annotations', metavar='path',
                    dest='annotations_path', default='dataset/',
                    help='Path to file which includes annotations e.g. \'banana/banana.csv\'')
parser.add_argument('-s','--save-path', metavar='path',
                    dest='save_path', default='dataset/',
                    help='Path to file which should be save e.g. \'banana/categories.csv\'')

args = parser.parse_args()

PATH_BASE = '/media/annaochjacob/crucial/'
PATH_ANNOTATIONS = PATH_BASE  + 'recorded_data/carla/' + args.annotations_path
PATH_SAVE = PATH_BASE  + 'recorded_data/carla/' + args.save_path

DELIMITER = ','
COMMENTS = ''
PRECISION = '%i'

CATEGORIES = {'s': 0,
             'ri': 1,
             'li': 2,
             'r' : 3,
             'l' : 4,
             'o' : 5,
             't' : 6}

def main():
    categories = get_categories(PATH_ANNOTATIONS)
    np.savetxt(PATH_SAVE, categories, delimiter=DELIMITER, header='category',
                comments=COMMENTS, fmt=PRECISION)

def get_categories(annotation_path):
    # Get annotations as (row, category) tuples
    df = pd.read_csv(annotation_path, delimiter=' ', header=None)
    annotations = [tuple(x) for x in df.values]
    n_frames = max(df[0]) + 1

    # Create array where index is frame index and value is category
    categories = np.zeros(n_frames)

    start_frame = 0
    for stop_frame, category in annotations:
        categories[start_frame:(stop_frame + 1)] = CATEGORIES[category]
        start_frame = stop_frame

    return categories

if __name__ == "__main__":
    main()
