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
    ''' Counts the points in each grid cell and returns it as a matrix.'''
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
    ''' Records the highest elevation within each grid cell.'''
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

def world_to_relative(x, y, yaw, w_coord):
    ''' Transform from world coordiantes into coordinates relative vehicle
    position and heading. '''
    # Shift locations so that location (x,y) in current frame is in origo
    w_coord[0] = w_coord[0] - x
    w_coord[1] = w_coord[1] - y

    # Rotate so heading of car in current frame is upwards
    theta = np.radians(-yaw + 90)
    c, s = np.cos(theta), np.sin(theta)
    R = [[c,-s],[s,c]]
    r_coord = np.dot(R, w_coord)

    return r_coord

def relative_to_world(x, y, yaw, r_coord):
    ''' Transform from coordiantes relative vehicle into world coordinates. '''
    # Rotate from heading upwards to heading in yaw direction
    theta = np.radians(yaw - 90)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c,-s), (s, c)))
    w_coord = np.dot(R, r_coord)

    # Shift locations back to current vehicle position
    w_coord[0] = w_coord[0] + x
    w_coord[1] = w_coord[1] + y
    return w_coord

def correct_carla_coordinates(x,y,yaw):
    ''' Carla uses flipped y (and thus also flipped yaw). This correction makes
    coordinates and yaw follow the right hand rule.'''
    x = x
    y = -y
    yaw = -yaw
    return x,y,yaw
