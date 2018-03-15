import os
import numpy as np
import argparse
from lidar_to_topview.main import lidar_to_topview, get_max_elevation
from intentions.carla.main import main as create_intentions
import time

parser = argparse.ArgumentParser(description='Create input and output files from recorded data')
parser.add_argument('--save-path', metavar='path',
                    dest='save_path', default='dataset/',
                    help='Where to save the dataset')
parser.add_argument('--data-path', metavar='path',
                    dest='data_path', default='recorded_data/',
                    help='Where to fetch the recorded data')
args = parser.parse_args()

# Define paths to data locations
ROI = 60
CELLS = 600

PATH_POINT_CLOUDS =  args.data_path + 'point_cloud/'
PATH_PLAYER = args.data_path + 'player_measurements/'
PATH_INTENTIONS = args.data_path + 'intentions/'
PATH_TRAFFIC = args.data_path + 'traffic_awareness/'

PATH_INPUT = args.save_path + 'input/'
PATH_OUTPUT = args.save_path + 'output/'

N_STEPS_FUTURE = 30
N_STEPS_PAST = 30
PRECISION = '%.8f'
DELIMITER = ','
COMMENTS = ''

def main():
    ''' Generates input and output files for player player_measurements.

    For each time step n, N_STEPS_PAST steps are gathered in one input file so that
    row k corresponds to time step (n-1-k). Row indices are assumed to begin with 0.
    For the output, N_STEPS_FUTURE steps are gathered similarly, where row k
    corresponds to time step (n+1+k).'''

    #First of all, create intentions and traffic awareness files.
    create_intentions(args)


    # Create directories
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists(PATH_INPUT):
        os.makedirs(PATH_INPUT)
    if not os.path.exists(PATH_INPUT + "values/"):
        os.makedirs(PATH_INPUT + "values/")
    if not os.path.exists(PATH_INPUT + "topviews/"):
        os.makedirs(PATH_INPUT + "topviews/")
    if not os.path.exists(PATH_INPUT + "topviews/max_elevation/"):
        os.makedirs(PATH_INPUT + "topviews/max_elevation/")
    if not os.path.exists(PATH_OUTPUT):
        os.makedirs(PATH_OUTPUT)

    # Player measurements matrix
    m_player = np.genfromtxt(PATH_PLAYER + 'pm.csv', delimiter=',', skip_header=True)
    m_intentions = np.genfromtxt(PATH_INTENTIONS + 'intentions.csv', delimiter =',', names=True)
    m_traffic = np.genfromtxt(PATH_TRAFFIC + 'traffic.csv', delimiter=',', names=True)
    n_frames = np.size(m_player,0)
    print('Found a total of %i frames' %n_frames)
    #m_traffic_csvreader = csv.reader()

    # for each frame in recorded data
    #for frame in range(0,n_frames):
    for frame in range(0,n_frames): #range(330,391):
        start = time.time()

        data_input = get_input(m_player, m_intentions, m_traffic, frame, n_frames, N_STEPS_PAST)
        data_output = get_output(m_player, frame, n_frames, N_STEPS_FUTURE)

        # Save information about past steps in  a separate csv file
        filename = (PATH_INPUT + 'values/input_%i.csv') %frame
        np.savetxt(filename, data_input, delimiter=DELIMITER, \
            header=get_input_header(), comments=COMMENTS, fmt=PRECISION)

        # Save information about past steps in  a separate csv file
        filename = (PATH_OUTPUT + 'output_%i.csv') %frame
        np.savetxt(filename, data_output, delimiter=DELIMITER, \
            header=get_output_header(), comments=COMMENTS, fmt=PRECISION)

        # TODO If we want, we can plot data input as an overlay on the topviews here

        filename = PATH_POINT_CLOUDS + 'pc_%i.csv' %frame
        point_cloud = point_cloud = np.loadtxt(filename, delimiter=',')

        # Save maximum elevation
        data_max_elevation = get_max_elevation(frame, point_cloud, ROI, CELLS)
        filename = (PATH_INPUT + 'topviews/max_elevation/me_%i.csv') %frame
        np.savetxt(filename, data_max_elevation, delimiter=DELIMITER, \
            comments=COMMENTS, fmt=PRECISION)

        # Save point count
        #data_count = lidar_to_topview('count', frame, point_cloud, ROI, CELLS)
        #filename = (PATH_INPUT + 'topviews/count/c_%i.csv') %frame
        #np.savetxt(filename, data_count, delimiter=DELIMITER, \
        #    comments=COMMENTS, fmt=PRECISION)

        stop = time.time()
        time_left = (stop - start) * (n_frames-frame)
        m, s = divmod(time_left, 60)
        h, m = divmod(m, 60)
        print("Frame %i - ETA %02d:%02d:%02d" % (frame, h, m, s))

def get_input(measurements, intentions, traffic, frame, n_frames, n_steps):
    all_inputs = np.zeros([n_steps, 11]) # Create container for past measurements
    x, y, yaw = measurements[frame,[2, 3, 11]]

    for past_step in range(0,n_steps):
        # Get index of past frames, i.e. exluding the current frame
        frame_index = frame - past_step - 1
        # If requested frame is further into the past than frame 0, use 0
        frame_index = max(frame_index,0)
        # Calculate relative location, forward acceleration etc.
        new_x, new_y = measurements[frame_index, [2, 3]]
        v_rel_x, v_rel_y = get_relative_location(x, y, yaw, new_x, new_y)
        acc_x, acc_y, acc_z = measurements[frame_index, [5, 6, 7]]
        v_forward_acceleration = get_forward_acceleration(acc_x, acc_y, acc_z)
        v_forward_speed = measurements[frame_index, 8]
        v_steer, v_throttle, v_break = measurements[frame_index, [17, 18, 19]]

        # Insert values in this frame's row
        frame_input = np.zeros(11)
        frame_input[0] = v_rel_x # location x relative to car
        frame_input[1] = v_rel_y # location y relative to car
        frame_input[2] = v_forward_acceleration # forward acceleration
        frame_input[3] = v_forward_speed # forward speed
        frame_input[4] = v_steer # steer
        frame_input[5] = intentions['intention_direction'][frame_index] # intention direction
        frame_input[6] = intentions['intention_proximity'][frame_index] # intention
        frame_input[7] = traffic['next_distance'][frame_index] # next_traffic_object_proximity
        frame_input[8] = traffic['current_speed_limit'][frame_index] # current_speed_limit
        frame_input[9] = traffic['next_speed_limit'][frame_index] # next_speed_limit (MIGHT BE NULL!)
        frame_input[10] = traffic['light_status'][frame_index] # traffic light status (MIGHT BE NULL!)


        all_inputs[past_step,:] = np.transpose(frame_input)

    return all_inputs

def get_output(measurements, frame, n_frames, n_steps):
    data_output = np.zeros([n_steps, 2])
    x, y, yaw = measurements[frame,[2, 3, 11]]

    for future_step in range(0,n_steps):
        # Get index of future frames, i.e. exluding the current frame
        frame_index = frame + future_step + 1
        # If requested frame is further into the future than last frame,
        # use last frame
        frame_index = min(frame_index, n_frames-1)
        new_x, new_y = measurements[frame_index, [2, 3]]
        rel_x, rel_y = get_relative_location(x, y, yaw, new_x, new_y)
        data_output[future_step,:] = [rel_x, rel_y]

    return data_output

def get_relative_location(x, y, yaw, new_x, new_y):
    # Rotate so heading of car in current frame is upwards
    theta = np.radians(-yaw - 90)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c,-s), (s, c)))
    x, y = np.dot(np.transpose([x, -y]) ,R) # Clockqise rotation when pos the
    new_x, new_y = np.dot(np.transpose([new_x, -new_y]) ,R)

    # Shift locations so that location in current frame (x,y) is in origo
    relative_x = new_x - x
    relative_y = new_y - y
    return relative_x, relative_y
    #return rel[0,0], rel[0,1]

def get_forward_acceleration(acc_x, acc_y, acc_z):
    squares = np.power([acc_x, acc_y, acc_z], 2)
    return np.sqrt(np.sum(squares))

def get_input_header():
    header = []
    header.append('location_x')
    header.append('location_y')
    header.append('forward_acceleration')
    header.append('forward_speed')
    header.append('steer')
    header.append('intention_direction')
    header.append('intention_proximity')
    header.append('next_distance')
    header.append('current_speed_limit')
    header.append('next_speed_limit')
    header.append('light_status')
    return DELIMITER.join(header)

def get_output_header():
    header = 'location_x,'
    header += 'location_y,'
    return header

def rotate(x, y, degrees):
    radians = np.deg2rad(degrees)
    c, s = np.cos(radians), np.sin(radians)
    r = np.matrix([[c, s], [-s, c]])
    return [x,y]*r

if __name__ == "__main__":
    main()
