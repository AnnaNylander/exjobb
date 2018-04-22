import os
import argparse
import shutil
import pandas as pd

''' Example:
Run the following line to copy files from fruit_salad/banana/ into
fruit_salad/banana_categorized/, where the annotations file is located at
recorded_data/banana/banana.csv

python3 categorize_dataset.py -c -s /fruit_salad/banana_categorized/ -d fruit_salad/banana/ -a recorded_data/banana/banana.csv
'''

parser = argparse.ArgumentParser(description='Split data into categories')

parser.add_argument('-a','--annotations', metavar='path',
                    dest='annotations_path', default='dataset/',
                    help='Path to file which includes annotations ex \'recorded_data/banana/banana.csv\'')
parser.add_argument('-s','--save-path', metavar='path',
                    dest='save_path', default='dataset/',
                    help='Foldername in /media/annaochjacob/crucial/dataset/ ex \'banana_split/\' (with trailing /)')
parser.add_argument('-d','--data-path', metavar='path',
                    dest='data_path', default='banana/',
                    help='Foldername in /media/annaochjacob/crucial/dataset/ ex \'banana/\' (with trailing /)')
parser.add_argument('-c','--copy', action='store_true',
                    dest='copy',
                    help='Copy data instead of moving')

args = parser.parse_args()

PATH_BASE = '/media/annaochjacob/crucial/'
PATH_BASE = '/Users/Jacob/Documents/Datasets/'

PATH_SAVE = PATH_BASE + args.save_path
PATH_DATA = PATH_BASE + args.data_path
ANNOTATION_PATH = PATH_BASE + args.annotations_path

CATEGORIES = {'s' : 'straight',
             'ri': 'right_intention',
             'li': 'left_intention',
             'r' : 'right',
             'l' : 'left',
             'o' : 'other',
             't' : 'traffic_light'}

def main():

    make_fake_set(PATH_BASE + 'fruit_salad/jacob_split/')

    # Create directories to move/copy into
    for key, name in CATEGORIES.items():
        make_dir(PATH_SAVE + '%s/input/values/' %name)
        make_dir(PATH_SAVE + '%s/input/topviews/max_elevation/' %name)
        make_dir(PATH_SAVE + '%s/output/' %name)

    # Get annotations as (row, category) tuples
    df = pd.read_csv(ANNOTATION_PATH, delimiter=' ', header=None)
    annotations = [tuple(x) for x in df.values]

    # Select transfer method, i.e. copy or move
    if args.copy:
        transfer = shutil.copy
    else:
        transfer = shutil.move

    # Go through all annotations
    frame = 0
    for stop_frame, category in annotations:
        # until stop_frame is reached, move files into correct category dir
        while frame <= stop_frame:
            src_values = PATH_DATA + 'input/values/input_%i.csv' %frame

            if os.path.isfile(src_values):
                src_me = PATH_DATA + 'input/topviews/max_elevation/me_%i.csv' %frame
                src_output = PATH_DATA + 'output/output_%i.csv' %frame

                cat_name = CATEGORIES[category]
                dst_values = PATH_SAVE + '%s/input/values/input_%i.csv' %(cat_name,frame)
                dst_me = PATH_SAVE + '%s/input/topviews/max_elevation/me_%i.csv' %(cat_name,frame)
                dst_output = PATH_SAVE + '%s/output/output_%i.csv' %(cat_name,frame)

                transfer(src_values, dst_values)
                transfer(src_me, dst_me)
                transfer(src_output, dst_output)

            else:
                print('Missing input for frame %i' %frame)

            frame += 1

    print('Dataset categorization completed.')

def make_dir(path):
    if not os.path.exists(path): os.makedirs(path)

def make_file(path):
    file = open(path, 'w')
    file.write('Hello World')
    file.close()

def make_fake_set(path_base):
    make_dir(path_base + 'input/values/')
    make_dir(path_base + 'input/topviews/max_elevation/')
    make_dir(path_base + 'output/')

    for i in range(100):
        make_file(path_base + 'input/values/input_%i.csv' %i)
        make_file(path_base + 'input/topviews/max_elevation/me_%i.csv' %i)
        make_file(path_base + 'output/output_%i.csv' %i)

if __name__ == "__main__":
    main()
