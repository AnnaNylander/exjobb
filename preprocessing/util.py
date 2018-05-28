import numpy as np
from scipy import misc
from PIL import Image

ELEVATION_MAX = 6
ELEVATION_MIN = -18

def trim_to_roi(point_cloud,roi):
    """ Remove points outside ROI."""
    inside_roi = np.max(np.absolute(point_cloud), axis=1) < roi/2
    return point_cloud[inside_roi]

def lidar_to_topview(point_cloud, roi, cells):
    grid = np.zeros([cells, cells, 1])

    for point in point_cloud:
        x, y, z = point
        x += roi/2
        x /= roi
        cell_x = (int) (x*cells) - 1

        y += roi/2
        y /= roi
        cell_y = (int) (y*cells) - 1

        grid[cell_x,cell_y] += 1

    grid = np.rot90(grid,1)
    return np.uint8(100*grid)

def get_max_elevation(frame, point_cloud, roi = 60, cells = 600):
    point_cloud = trim_to_roi(point_cloud,roi) # Always trim to roi first
    grid = np.full([cells, cells, 1], np.nan)

    for point in point_cloud:
        x, y, z = point
        x += roi/2
        x /= roi
        cell_x = (int) (x*cells) - 1

        y += roi/2
        y /= roi
        cell_y = (int) (y*cells) - 1

        element = grid[cell_x,cell_y]
        if np.isnan(element) or element < z:
             grid[cell_x,cell_y] = z

    grid = np.nan_to_num(grid) # Replace all nans with zeros
    grid = grid/(ELEVATION_MAX - ELEVATION_MIN)
    grid = grid + 0.5
    grid = np.rot90(grid,1)
    return np.uint8(grid*255)
    #return np.uint8(100*grid) #enhance contrast
    #img = Image.fromarray(grid)
    #img = img.rotate(180)
    #img.show()
    #img.save('output/topview_%i.png' %frame)
