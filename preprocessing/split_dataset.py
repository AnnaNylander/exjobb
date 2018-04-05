import os
import numpy as np
import argparse
import shutil

parser = argparse.ArgumentParser(description='Split data set in training, testing, and validation set')
parser.add_argument('--train', metavar='ratio of dataset', type=float,
                    dest='ratio_train', default=0.7,
                    help='Ratio of original data set to put in training set. (default 0.7)')
parser.add_argument('--test', metavar='ratio of dataset',type=float,
                    dest='ratio_test', default=0.2,
                    help='Ratio of original data set to put in training set. (default 0.2)')
parser.add_argument('-s','--save-path', metavar='path',
                    dest='save_path', default='dataset/',
                    help='Foldername in /media/annaochjacob/crucial/dataset/ ex \'banana_split/\' (with trailing /)')
parser.add_argument('-d','--data-path', metavar='path',
                    dest='data_path', default='recorded_data/',
                    help='Foldername in /media/annaochjacob/crucial/dataset/ ex \'banana/\' (with trailing /)')
parser.add_argument('--copy', action='store_true',
                    dest='copy',
                    help='Copy data instead of moving')

args = parser.parse_args()

PATH_BASE = '/media/annaochjacob/crucial/'

SAVE_PATH =  PATH_BASE + 'dataset/' + args.save_path
DATA_PATH = PATH_BASE +'dataset/' +  args.data_path

def main():
    '''Splits a dataset into train, test and validate subsets. Train and test
    ratios are set, validate will consist of the remaining data.
    '''
    # Create directories
    if not os.path.exists(SAVE_PATH + '/train'):
        os.makedirs(SAVE_PATH + '/train/input/values')
        os.makedirs(SAVE_PATH + '/train/input/topviews/max_elevation')
        os.makedirs(SAVE_PATH + '/train/output')
    if not os.path.exists(SAVE_PATH + '/test'):
        os.makedirs(SAVE_PATH + '/test/input/values')
        os.makedirs(SAVE_PATH + '/test/input/topviews/max_elevation')
        os.makedirs(SAVE_PATH + '/test/output')
    if not os.path.exists(SAVE_PATH + '/validate'):
        os.makedirs(SAVE_PATH + '/validate/input/values')
        os.makedirs(SAVE_PATH + '/validate/input/topviews/max_elevation')
        os.makedirs(SAVE_PATH + '/validate/output')

    # Get sorted filenames
    input_values = sorted(os.listdir(DATA_PATH + '/input/values'))
    input_topviews = sorted(os.listdir(DATA_PATH + '/input/topviews/max_elevation'))
    output = sorted(os.listdir(DATA_PATH + '/output'))

    # Get number of examples in each subset
    n_examples = len(input_values)
    n_train = int(args.ratio_train*n_examples)
    n_test = int(args.ratio_test*n_examples)
    n_validate = n_examples - (n_train + n_test)

    indices = np.arange(n_examples)
    np.random.shuffle(indices)
    ind_train, ind_test, ind_validate = np.split(indices,[n_train,n_train+n_test])

    # move or copy files
    for i in ind_train:
        src = DATA_PATH + '/input/values/' + input_values[i]
        dst = SAVE_PATH + '/train/input/values/' + input_values[i]
        shutil.copy(src, dst) if args.copy else shutil.move(src, dst)

        src = DATA_PATH + '/input/topviews/max_elevation/' + input_topviews[i]
        dst = SAVE_PATH + '/train/input/topviews/max_elevation/' + input_topviews[i]
        shutil.copy(src, dst) if args.copy else shutil.move(src, dst)

        src = DATA_PATH + '/output/' + output[i]
        dst = SAVE_PATH + '/train/output/' + output[i]
        shutil.copy(src, dst) if args.copy else shutil.move(src, dst)

    for i in ind_test:
        src = DATA_PATH + '/input/values/' + input_values[i]
        dst = SAVE_PATH + '/test/input/values/' + input_values[i]
        shutil.copy(src, dst) if args.copy else shutil.move(src, dst)

        src = DATA_PATH + '/input/topviews/max_elevation/' + input_topviews[i]
        dst = SAVE_PATH + '/test/input/topviews/max_elevation/' + input_topviews[i]
        shutil.copy(src, dst) if args.copy else shutil.move(src, dst)

        src = DATA_PATH + '/output/' + output[i]
        dst = SAVE_PATH + '/test/output/' + output[i]
        shutil.copy(src, dst) if args.copy else shutil.move(src, dst)

    for i in ind_validate:
        src = DATA_PATH + '/input/values/' + input_values[i]
        dst = SAVE_PATH + '/validate/input/values/' + input_values[i]
        shutil.copy(src, dst) if args.copy else shutil.move(src, dst)

        src = DATA_PATH + '/input/topviews/max_elevation/' + input_topviews[i]
        dst = SAVE_PATH + '/validate/input/topviews/max_elevation/' + input_topviews[i]
        shutil.copy(src, dst) if args.copy else shutil.move(src, dst)

        src = DATA_PATH + '/output/' + output[i]
        dst = SAVE_PATH + '/validate/output/' + output[i]
        shutil.copy(src, dst) if args.copy else shutil.move(src, dst)

if __name__ == "__main__":
    main()
