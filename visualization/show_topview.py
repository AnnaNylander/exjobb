import numpy as np
from PIL import Image
import argparse

DATA_PATH = '/media/annaochjacob/crucial/datasets/test_set1/input/topviews/'

parser = argparse.ArgumentParser(description='Show topview')
parser.add_argument('--step', metavar='integer', type=int,
                    dest='timeStep', default='0',
                    help='Time step (frame index) to display topview for.')
parser.add_argument('--statistic', metavar='integer', type=int,
                    dest='statistic', default='0',
                    help='Statistic to show')

args = parser.parse_args()

filename = DATA_PATH
if args.statistic == 0:
    filename += 'max_elevation/me_%i.csv' %args.timeStep
elif args.statistic == 1:
    filename += 'count/c_%i.csv' %args.timeStep

grid = np.genfromtxt(filename, delimiter=',')
img = Image.fromarray(grid)
img = img.rotate(180);
#img.save('my.png')
img.show()
