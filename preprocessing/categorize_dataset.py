import os
import numpy as np
import argparse
import shutil
import re

PATH_BASE = '/Users/Jacob/Documents/Datasets/fruit_salad/jacob_split/'
ANNOTATION_PATH = '/Users/Jacob/Documents/Datasets/fruit_salad/apple.csv'

STRAIGHT = '0'
INTENTION_R = '1'
INTENTION_L = '2'
RIGHT = '3'
LEFT = '4'
OTHER = '5'
TRAFFIC_LIGHT = '6'

CATEGORIES = [STRAIGHT, INTENTION_R, INTENTION_L,
                RIGHT, LEFT, OTHER, TRAFFIC_LIGHT]

def main():

    make_fake_set(PATH_BASE)

    # Create directories
    for subset in ['train/', 'test/', 'validate/']:
        for c in CATEGORIES:
            make_dir(PATH_BASE + '%s%s/input/values/' %(subset, c))
            make_dir(PATH_BASE + '%s%s/input/topviews/max_elevation/' %(subset, c))
            make_dir(PATH_BASE + '%s%s/output/' %(subset, c))

    annotations = np.genfromtxt(ANNOTATION_PATH)
    last_frame = int(annotations[-1,0])

    # for each row in the annotations table
    frame = 0
    for i_row in range(np.size(annotations, 0)):
        stop_frame, category = annotations[i_row]
        category = int(category)

        # until stop_frame is reached, move files into correct category dir
        while frame <= stop_frame:

            for subset in ['train/', 'test/', 'validate/']:
                src_values = PATH_BASE + subset + 'input/values/input_%i.csv' %frame

                if os.path.isfile(src_values):
                    src_me = PATH_BASE + subset + 'input/topviews/max_elevation/me_%i.csv' %frame
                    src_output = PATH_BASE + subset + 'output/output_%i.csv' %frame

                    dst_values = PATH_BASE + subset + '%s/input/values/input_%i.csv' %(category,frame)
                    dst_me = PATH_BASE + subset + '%s/input/topviews/max_elevation/me_%i.csv' %(category,frame)
                    dst_output = PATH_BASE + subset + '%s/output/output_%i.csv' %(category,frame)

                    shutil.move(src_values, dst_values)
                    shutil.move(src_me, dst_me)
                    shutil.move(src_output, dst_output)

            frame += 1

def make_dir(path):
    if not os.path.exists(path): os.makedirs(path)

def make_file(path):
    file = open(path, 'w')
    file.write('Hello World')
    file.close()

def make_fake_set(path_base):
    for subset in ['train/', 'test/', 'validate/']:
        make_dir(path_base + subset + 'input/values/')
        make_dir(path_base + subset + 'input/topviews/max_elevation/')
        make_dir(path_base + subset + 'output/')

        for i in range(100):
            make_file(path_base + subset + 'input/values/input_%i.csv' %i)
            make_file(path_base + subset + 'input/topviews/max_elevation/me_%i.csv' %i)
            make_file(path_base + subset + 'output/output_%i.csv' %i)

if __name__ == "__main__":
    main()
