import os
import numpy as np

# Define paths to data locations
PATH_DATA = '/Users/Jacob/Documents/Datasets/exjobb/recorded_data_2018-03-02'
PATH_POINT_CLOUDS = PATH_DATA + '/point_cloud'
PATH_PLAYER = PATH_DATA + '/player_measurements/pm.csv'
PATH_STATIC = PATH_DATA + '/static_measurements'
PATH_DYNAMIC = PATH_DATA + '/dynamic_measurements'
PATH_SAVE = PATH_DATA + '/data_set'
PATH_INPUT = PATH_SAVE + '/input'
PATH_OUTPUT = PATH_SAVE + '/output'
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

    # Player measurements matrix
    m_player = np.genfromtxt(PATH_PLAYER, delimiter=' ', skip_header=True)
    n_frames = np.size(m_player,0)

    # for each frame in recorded data
    for frame in range(0,n_frames):
        #get_input_data(frame, x, y ,yaw, N_STEPS_PAST)
        # Get x, y and yaw for current frame to make past and future positions
        # relative to the current position.
        x, y, yaw = m_player[frame,[2, 3, 11]]

        data_input = get_input(m_player, frame, n_frames, N_STEPS_PAST, x, y, yaw)
        data_output = get_output(m_player, frame, n_frames, N_STEPS_FUTURE, x, y, yaw)

        # Save information about past steps in  a separate csv file
        filename_input = (PATH_INPUT + '/input_%i.csv') %frame
        np.savetxt(filename_input, data_input, delimiter=DELIMITER, \
            header=get_input_header(), comments=COMMENTS, fmt=PRECISION)

        # Save information about past steps in  a separate csv file
        filename_output = (PATH_OUTPUT + '/output_%i.csv') %frame
        np.savetxt(filename_output, data_output, delimiter=DELIMITER, \
            header=get_output_header(), comments=COMMENTS, fmt=PRECISION)


def get_input(measurements, frame, n_frames, n_steps, x, y, yaw):
    all_inputs = np.zeros([n_steps, 7])

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

        frame_input = np.zeros(7)
        frame_input[0] = v_rel_x # location x relative to car
        frame_input[1] = v_rel_y # location y relative to car
        frame_input[2] = v_forward_acceleration # forward acceleration
        frame_input[3] = v_forward_speed # forward speed
        frame_input[4] = v_steer # steer
        #frame_input[5] = v_throttle # throttle
        #frame_input[6] = v_break # break
        frame_input[5] = 0 # intention direction
        frame_input[6] = 0 # intention proximity

        all_inputs[past_step,:] = np.transpose(frame_input)

    return all_inputs

def get_output(measurements, frame, n_frames, n_steps, x, y, yaw):
    data_output = np.zeros([n_steps, 2])

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
    rel_x = new_x - x
    rel_y = new_y - y
    # Rotate so heading of car in frame t is upwards
    rel = rotate(rel_x, rel_y, -np.sign(yaw)*yaw + 90)


    return rel[0,0], rel[0,1]

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
    header.append('throttle')
    header.append('brake')
    header.append('intention_direction')
    header.append('intention_proximity')
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
