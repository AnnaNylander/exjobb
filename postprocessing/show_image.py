import numpy as np
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description='Create input and output files from recorded data')
parser.add_argument('--path', metavar='file.csv',
                    dest='path', default='/media/annaochjacob/crucial/recorded_data/carla/recorded_data_2018-03-07/point_cloud/',
                    help='Path to folder where images is stored.')
parser.add_argument('-i', '--index', default=0, type=int,
                    metavar='N', help='index of pointcloud.')
args = parser.parse_args()

CELLS = 600
ROI = 60

point_cloud = np.genfromtxt(args.path + 'pc_%i.csv' %args.index, delimiter=',')

grid = np.zeros([CELLS,CELLS])

for point in point_cloud:
    x, y, z = point
    x += ROI/2
    x /= ROI
    cell_x = int(x*CELLS)

    y += ROI/2
    y /= ROI
    cell_y = int(y*CELLS)

    grid[cell_x,cell_y] += 1

grid = 64*grid

img = Image.fromarray(grid)
img = img.rotate(180);
#img.save('my.png')
img.show()
