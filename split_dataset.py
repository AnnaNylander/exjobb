import os
import numpy as np
import argparse
import shutil

parser = argparse.ArgumentParser(description='Split data set in training, testing, and validation set')
parser.add_argument('--train', metavar='ratio of dataset', type=float,
                    dest='ratio_train', default=0.7,
                    help='Ratio of original data set to put in training set')
parser.add_argument('--test', metavar='ratio of dataset',type=float,
                    dest='ratio_test', default=0.2,
                    help='Ratio of original data set to put in training set')
parser.add_argument('--save-path', metavar='path',
                    dest='SAVE_PATH', default='dataset_split',
                    help='Where to save the split dataset')
parser.add_argument('--data-path', metavar='path',
                    dest='DATA_PATH', default='recorded_data',
                    help='Where to fetch the dataset')
parser.add_argument('--copy', metavar='boolean',
                    dest='copy', default=False,
                    help='Copy data instead of moving')

args = parser.parse_args()


def main():
    '''Splits a dataset into train, test and validate subsets. Train and test
    ratios are set, validate will consist of the remaining data.
    '''
    # Create directories
    if not os.path.exists(args.SAVE_PATH + '/train'):
        os.makedirs(args.SAVE_PATH + '/train/input/values')
        os.makedirs(args.SAVE_PATH + '/train/input/topviews/max_elevation')
        os.makedirs(args.SAVE_PATH + '/train/output')
    if not os.path.exists(args.SAVE_PATH + '/test'):
        os.makedirs(args.SAVE_PATH + '/test/input/values')
        os.makedirs(args.SAVE_PATH + '/test/input/topviews/max_elevation')
        os.makedirs(args.SAVE_PATH + '/test/output')
    if not os.path.exists(args.SAVE_PATH + '/validate'):
        os.makedirs(args.SAVE_PATH + '/validate/input/values')
        os.makedirs(args.SAVE_PATH + '/validate/input/topviews/max_elevation')
        os.makedirs(args.SAVE_PATH + '/validate/output')

    # Get sorted filenames
    input_values = sorted(os.listdir(args.DATA_PATH + '/input/values'))
    input_topviews = sorted(os.listdir(args.DATA_PATH + '/input/topviews/max_elevation'))
    output = sorted(os.listdir(args.DATA_PATH + '/output'))

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
        src = args.DATA_PATH + '/input/values/' + input_values[i]
        dst = args.SAVE_PATH + '/train/input/values/' + input_values[i]
        shutil.copy(src, dst) if args.copy else shutil.move(src, dst)

        src = args.DATA_PATH + '/input/topviews/max_elevation/' + input_topviews[i]
        dst = args.SAVE_PATH + '/train/input/topviews/max_elevation/' + input_topviews[i]
        shutil.copy(src, dst) if args.copy else shutil.move(src, dst)

        src = args.DATA_PATH + '/output/' + output[i]
        dst = args.SAVE_PATH + '/train/output/' + output[i]
        shutil.copy(src, dst) if args.copy else shutil.move(src, dst)

    for i in ind_test:
        src = args.DATA_PATH + '/input/values/' + input_values[i]
        dst = args.SAVE_PATH + '/test/input/values/' + input_values[i]
        shutil.copy(src, dst) if args.copy else shutil.move(src, dst)

        src = args.DATA_PATH + '/input/topviews/max_elevation/' + input_topviews[i]
        dst = args.SAVE_PATH + '/test/input/topviews/max_elevation/' + input_topviews[i]
        shutil.copy(src, dst) if args.copy else shutil.move(src, dst)

        src = args.DATA_PATH + '/output/' + output[i]
        dst = args.SAVE_PATH + '/test/output/' + output[i]
        shutil.copy(src, dst) if args.copy else shutil.move(src, dst)

    for i in ind_validate:
        src = args.DATA_PATH + '/input/values/' + input_values[i]
        dst = args.SAVE_PATH + '/validate/input/values/' + input_values[i]
        shutil.copy(src, dst) if args.copy else shutil.move(src, dst)

        src = args.DATA_PATH + '/input/topviews/max_elevation/' + input_topviews[i]
        dst = args.SAVE_PATH + '/validate/input/topviews/max_elevation/' + input_topviews[i]
        shutil.copy(src, dst) if args.copy else shutil.move(src, dst)

        src = args.DATA_PATH + '/output/' + output[i]
        dst = args.SAVE_PATH + '/validate/output/' + output[i]
        shutil.copy(src, dst) if args.copy else shutil.move(src, dst)

if __name__ == "__main__":
    main()
