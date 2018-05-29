import os
import numpy as np
import argparse
from util import get_max_elevation
import intentions.carla.create_intentions as create_intentions
import time

parser = argparse.ArgumentParser(description='Create input and output files from recorded data')
parser.add_argument('-s','--save-path', metavar='path',
                    dest='save_path', default='dataset/',
                    help='Foldername in /media/annaochjacob/crucial/dataset/ ex \'flexible/train/apple/\' (with trailing /)')
parser.add_argument('-d','--data-path', metavar='path',
                    dest='data_path', default='recorded_data/',
                    help='Foldername in /media/annaochjacob/crucial/recorded_data/carla/ ex \'train/apple/\' (with trailing /)')
parser.add_argument('-c', dest='include_categories', action='store_true',
                    help='Adds category column to output file if true. Default False.')
parser.add_argument('-m', dest='create_max_elevation', action='store_true',
                    help='Creates max elevation files if true. Default False.')
args = parser.parse_args()

ROI = 60    # Region of interest side length of square, in meters
CELLS = 600 # The number number of cells along the side of the topview

# Define paths to data locations
PATH_BASE = '/media/annaochjacob/crucial/'
PATH_DATA = PATH_BASE + 'recorded_data/carla/' + args.data_path
PATH_POINT_CLOUDS =  PATH_DATA + 'point_cloud/'
PATH_PLAYER = PATH_DATA + 'player_measurements/'
PATH_INTENTIONS = PATH_DATA + 'intentions/'
PATH_TRAFFIC = PATH_DATA + 'traffic_awareness/'
PATH_CATEGORIES = PATH_DATA + 'categories/'
PATH_SAVE = PATH_BASE + 'dataset/'+ args.save_path

# Set csv file attribute values
PRECISION = '%.8f'
DELIMITER = ','
COMMENTS = ''

def main():
    ''' Generates topview files, category and intention files for player
        player_measurements.'''

    #First of all, create intentions and traffic awareness files.
    create_intentions.create_intentions(args)

    # Create directories
    if not os.path.exists(PATH_SAVE):
        os.makedirs(PATH_SAVE)
    if not os.path.exists(PATH_SAVE + "max_elevation/"):
        os.makedirs(PATH_SAVE + "max_elevation/")

    # Player measurements matrix
    m_player = np.genfromtxt(PATH_PLAYER + 'pm.csv', delimiter=',', skip_header=True)
    m_intentions = np.genfromtxt(PATH_INTENTIONS + 'intentions.csv', delimiter =',', names=True)
    m_traffic = np.genfromtxt(PATH_TRAFFIC + 'traffic.csv', delimiter=',', names=True)

    # Categories e.g. left, right, left with intention, are a special case, since
    # only the train set makes use of categories.
    if args.include_categories:
        m_categories = np.genfromtxt(PATH_CATEGORIES + 'categories.csv',
                                        delimiter=',', skip_header=True)
    else:
        m_categories = None
    n_frames = np.size(m_player,0)
    print('Found a total of %i frames' %n_frames)

    #---------------------------------------------------------------------------
    # Select the interesting values and merge into one matrix
    values = np.zeros([n_frames, 13])
    for frame_index in range(0,n_frames):
        values[frame_index, :] = get_values(frame_index, m_player, m_intentions,
                                                m_traffic, m_categories)

    # Save value matrix
    filename = (PATH_SAVE + 'values.csv')
    np.savetxt(filename, values, delimiter=DELIMITER, header=get_header(),
                comments=COMMENTS, fmt=PRECISION)

    #---------------------------------------------------------------------------
    # Create max elevation images if desired
    if not args.create_max_elevation:
        return

    for frame_index in range(0,n_frames):
        if frame_index % 100 == 0:
            print('Max elevation frame %i' %frame_index)
        pc_path = PATH_POINT_CLOUDS + 'pc_%i.csv' %frame_index
        point_cloud = np.genfromtxt(pc_path, delimiter=',', skip_header=True)
        max_elevation = get_max_elevation(frame_index, point_cloud, ROI, CELLS)
        #TODO move this squeese into get_max_elevation?
        max_elevation = np.squeeze(max_elevation, 2)
        filename = (PATH_SAVE + 'max_elevation/me_%i.csv') %frame_index
        np.savetxt(filename, max_elevation, delimiter=DELIMITER,
                    comments=COMMENTS, fmt='%u')

def get_header():
    header = []
    header.append('location_x')
    header.append('location_y')
    header.append('yaw')
    header.append('total_acceleration')
    header.append('forward_speed')
    header.append('steer')
    header.append('intention_direction')
    header.append('intention_proximity')
    header.append('next_distance')
    header.append('current_speed_limit')
    header.append('next_speed_limit')
    header.append('light_status')
    header.append('category')
    return DELIMITER.join(header)

def get_values(frame_index, m_player, m_intentions, m_traffic, m_categories=None):
    ''' Returns the values of interest in a given frame.
        Note that the carla coordinate system correction is made here.'''

    values = np.zeros(13) # Init matrix holding all values in given frame

    x, y, yaw = m_player[frame_index, [2, 3, 11]]
    x, y, yaw = correct_carla_coordinates(x, y, yaw)

    acc_x, acc_y, acc_z = m_player[frame_index, [5, 6, 7]]
    total_acceleration = get_total_acceleration(acc_x, acc_y, acc_z)
    forward_speed = m_player[frame_index, 8]
    steer, throttle, brake = m_player[frame_index, [17, 18, 19]]

    values[0] = x
    values[1] = y
    values[2] = yaw
    values[3] = total_acceleration
    values[4] = forward_speed
    values[5] = steer
    values[6] = m_intentions['intention_direction'][frame_index]
    values[7] = m_intentions['intention_proximity'][frame_index]
    values[8] = m_traffic['next_distance'][frame_index]
    values[9] = m_traffic['current_speed_limit'][frame_index]
    values[10] = m_traffic['next_speed_limit'][frame_index] # (TODO fix MIGHT BE NULL!)
    values[11] = m_traffic['light_status'][frame_index] # (TODO fix MIGHT BE NULL!)
    if m_categories is not None:
        values[12] = m_categories[frame_index]
    return values

def correct_carla_coordinates(x,y,yaw):
    x = x
    y = -y
    yaw = -yaw
    return x,y,yaw

def get_total_acceleration(acc_x, acc_y, acc_z):
    squares = np.power([acc_x, acc_y, acc_z], 2)
    return np.sqrt(np.sum(squares))

if __name__ == "__main__":
    main()
