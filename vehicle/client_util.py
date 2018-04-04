import numpy as np
import matplotlib.pyplot as plt
import math
import os

ELEVATION_MAX = 6
ELEVATION_MIN = -18
RADIUS=3

def main():
    # Define two points and the yaw in the first point
    x, y = 202.55, 55.84
    new_x, new_y = 180.90164062, 55.83779297
    yaw = 179.99975586

    # Example with multiple points
    #x = 0
    #y = 0
    #new_x = -np.arange(30,0,-1)
    #new_y = -np.arange(30,0,-1)

    # Check that x and y are translated into origo
    o_x, o_y = world_to_relative(x, -y, yaw, x, -y)
    # Transform another point into relative coordinates
    rel_x, rel_y = world_to_relative(x, -y, yaw, new_x, -new_y)
    # Transform the point back into world coordinates
    back_x, back_y = relative_to_world(x, -y, yaw, rel_x, rel_y)

    # plot in world coordinates
    plt.figure()
    plt.scatter(x, -y, marker='o', c='r')
    plt.scatter(new_x, -new_y, marker='.', c='b')
    plt.scatter(back_x, back_y, marker='x', c='g')

    # plot in relative coordinates
    plt.figure()
    plt.scatter(o_x, o_y, marker='o', c='r')
    plt.scatter(rel_x, rel_y, marker='.', c='b')
    plt.show()

def world_to_relative(x, y, yaw, new_x, new_y):
    # Rotate so heading of car in current frame is upwards
    theta = np.radians(-yaw - 90)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c,-s), (s, c)))
    #x, y = np.dot(np.transpose([x, y]) ,R) # Clockwise rotation when pos theta
    x, y = np.dot(np.transpose([x, y]) ,R) # Clockwise rotation when pos theta
    new_x, new_y = np.hsplit(np.dot(np.transpose([new_x, new_y]) ,R), 2)

    # Shift locations so that location in current frame (x,y) is in origo
    relative_x = new_x - x
    relative_y = new_y - y
    #print(x,new_x, relative_x)
    return relative_x, relative_y

def relative_to_world(x, y, yaw, rel_x, rel_y):
    # Rotate from heading upwards to heading in yaw dire
    theta = np.radians(yaw + 90)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c,-s), (s, c)))

    #print(np.dot(np.transpose([rel_x, rel_y]) ,R).squeeze()
    rel_x, rel_y = np.hsplit(np.dot(np.transpose([rel_x, rel_y]) ,R).squeeze(), 2)

    #print(rel_x)
    #print(rel_y)

    world_x = rel_x + x
    world_y = rel_y + y
    return world_x, world_y

def get_max_elevation(point_cloud, ROI = 60, CELLS = 600):
    grid = np.full([CELLS,CELLS,1], np.nan)

    for point in point_cloud:
        x, y, z = point
        x += ROI/2
        x /= ROI
        cell_x = (int) (x*CELLS) - 1

        y += ROI/2
        y /= ROI
        cell_y = (int) (y*CELLS) - 1

        element = grid[cell_x,cell_y]
        if np.isnan(element) or element < z:
             grid[cell_x,cell_y] = z

    grid = np.nan_to_num(grid) # Replace all nans with zeros
    grid = grid/(ELEVATION_MAX - ELEVATION_MIN)
    grid = grid + 0.5
    grid = np.rot90(grid,2)
    return np.uint8(grid*255)

def get_input(measurements, intentions, traffic, frame, n_steps):
    all_inputs = np.zeros([n_steps, 11]) # Create container for past measurements
    x, y, yaw = measurements[frame,[2, 3, 11]]

    for past_step in range(0,n_steps):
        # Get index of past frames, i.e. exluding the current frame
        frame_index = frame - past_step - 1
        # If requested frame is further into the past than frame 0, use 0
        frame_index = max(frame_index,0)
        # Calculate relative location, forward acceleration etc.
        new_x, new_y = measurements[frame_index, [2, 3]]
        # Notice the minus signs on y and new_y because of carla's world axes!
        v_rel_x, v_rel_y = world_to_relative(x, -y, yaw, new_x, -new_y)
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
        frame_input[5] = intentions[frame_index][0] # intention direction
        frame_input[6] = intentions[frame_index][1] # intention
        frame_input[7] = traffic[frame_index][0] # next_traffic_object_proximity
        frame_input[8] = traffic[frame_index][1] # current_speed_limit
        frame_input[9] = np.nan_to_num(traffic[frame_index][2]) # next_speed_limit (MIGHT BE NULL!)
        frame_input[10] = np.nan_to_num(traffic[frame_index][3]) # traffic light status (MIGHT BE NULL!)

        all_inputs[past_step,:] = np.transpose(frame_input)

    return all_inputs

def get_forward_acceleration(acc_x, acc_y, acc_z):
    squares = np.power([acc_x, acc_y, acc_z], 2)
    return np.sqrt(np.sum(squares))

def GetIntention(player_measurements, prev_intentions, intentions_path):
    """used for turn intentions"""
    next_value_flag = False
    if intentions_path.size == 0:
        return [0, 100], next_value_flag #if path is empy, turn nowhere in 100 meters. (default values)

    next_pos = (intentions_path['location_x'][0],intentions_path['location_y'][0])
    next_value = prev_intentions[0][0] #direction in col 0
    next_distance = prev_intentions[0][1] # distance in col 1
    last_position = (player_measurements[-2][2],player_measurements[-2][3]) # col 2 and 3
    current_position = (player_measurements[-1][2],player_measurements[-1][3])

    if (next_pos == None or isWithinRadius(current_position,next_pos,RADIUS)):
        next_pos, next_value = (intentions_path['location_x'][1],intentions_path['location_y'][1])
        next_value = intentions_path['intention_direction'][1]
        next_value_flag = True
        next_distance = intentions_path['next_distance'][1]

    # get updated turn proximity
    next_distance = next_distance - getEulerDistance(last_position, current_position)
    next_distance = max(next_distance, 0)

    return [next_value, next_distance], next_value_flag #if got next intention, reutnr true so reciever can relete row.

def GetTraffic(player_measurements, static_measurements, dynamic_measurements, prev_traffic, traffic_path):
    """use for traffic"""
    next_value_flag = False
    if len(traffic_path['id']) == 0:
        return [0, prev_traffic[0][1], 0, 0], next_value_flag #if path is empy, return some default values

    next_pos = (traffic_path['location_x'][0],traffic_path['location_y'][0])
    next_id = traffic_path['id'][0]
    next_distance = prev_traffic[0][0] # distance in col 0
    last_position = (player_measurements[-2][2],player_measurements[-2][3]) # col 2 and 3
    current_position = (player_measurements[-1][2],player_measurements[-1][3])
    isTrafficLight = True if static_measurements[next_id]['type'] == 3 else False
    current_speed_limit = prev_traffic[0][1]
    next_speed_limit = prev_traffic[0][2]

    if (next_pos == None or isWithinRadius(current_position,next_pos,RADIUS)):
        current_speed_limit = next_speed_limit #can this ever be none?
        next_pos = (traffic_path['location_x'][1],traffic_path['location_y'][1])
        next_id = traffic_path['id'][1]
        next_value_flag = True
        next_distance = traffic_path['next_distance'][1]
        isTrafficLight = True if static_measurements[next_id]['type'] == 3 else False

    # get updated turn proximity
    next_distance = next_distance - getEulerDistance(last_position, current_position)
    next_distance = max(next_distance, 0)

    light_status = None
    next_speed_limit = None
    if isTrafficLight:
        light_status = dynamic_measurements[next_id][-1] #get current status for traffic light
    else:
        next_speed_limit = static_measurements[next_id]['speed_limit']

    return [next_distance, current_speed_limit, next_speed_limit, light_status], next_value_flag

def isWithinRadius(a, b, r = 3):
    if getEulerDistance(a,b) < r:
        return True
    return False

def getEulerDistance(pos_a, pos_b):
    (a_lat,a_lon) = pos_a
    (b_lat,b_lon) = pos_b
    dist = math.sqrt((b_lat - a_lat)**2 + (b_lon - a_lon)**2)
    return dist


def generate_output(frame, output, path):
    if not os.path.exists(path):
        os.makedirs(path)

    output = np.transpose(output)
    filename = path + 'gen_%i.csv' %(frame)
    np.savetxt(filename, output, comments='', delimiter=',',fmt='%.8f',
                    header='x,y')

def find_traffic_sign_ids(static_measurements, traffic_path):
    updated_traffic_path = {'id':[], 'location_x':[], 'location_y':[],'next_distance':[]}

    for idx, sign in enumerate(traffic_path):
        annotation_location = (sign[0], sign[1])
        for row in static_measurements:
            sign_location = (static_measurements[row]['location_x'], static_measurements[row]['location_y'])
            if isWithinRadius(annotation_location, sign_location, r = 2):
                updated_traffic_path['id'].append(row)
                updated_traffic_path['location_x'].append(sign[0])
                updated_traffic_path['location_y'].append(sign[1])
                updated_traffic_path['next_distance'].append(sign[2])
                break
    if not len(updated_traffic_path['id']) == len(traffic_path):
        print("COULD NOT LOCATE ALL TRAFFIC LIGHTS/SIGNS. PLEASE CHECK ANNOTATIONS.")
        print(traffic_path)
        print(updated_traffic_path)
    return updated_traffic_path


if __name__ == '__main__':
    main()
