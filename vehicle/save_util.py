import numpy as np
from preprocessing.point_cloud_to_image import trim_to_roi
import datetime

ROI = 60        # Region of interest side length, in meters.
FAR = 1000      # Far plane distance
POINT_CLOUD_PRECISION = '%.3f'
MEASUREMENTS_PRECISION = '%.8f'
DELIMITER = ','
COMMENTS = ''

def save_point_cloud(frame, point_cloud, save_path):
    filename = save_path + 'pc_%i.csv' %frame

    # Reverse y and z because of carla
    pc = [[x,-y,-z] for [x,y,z] in point_cloud]

    # format point cloud
    pc = '\n'.join(['{:.2f},{:.2f},{:.2f}'.format(*p) for p in pc])

    # Add header
    pc = '\n'.join(['x,y,z', pc])

    # Open the file and save point cloud in it
    with open(filename, 'w+') as csv_file:
        csv_file.write(pc)

def save_player_measurements(measurements, save_path):
    # Save measurements of whole episode to one file
    header_player = get_player_measurements_header()
    np.savetxt(save_path + "/pm.csv", \
        measurements, fmt=MEASUREMENTS_PRECISION, header=header_player, \
        comments=COMMENTS, delimiter=DELIMITER)

def get_player_measurements(measurements):
    # Separate measurement types
    player = measurements.player_measurements
    control = player.autopilot_control
    transform = player.transform
    acceleration = player.acceleration

    player_values = np.zeros(22)

    # Time
    player_values[0] = measurements.platform_timestamp
    player_values[1] = measurements.game_timestamp

    # Location
    player_values[2] = transform.location.x
    player_values[3] = transform.location.y
    player_values[4] = transform.location.z

    # Acceleration and forward speed
    player_values[5] = acceleration.x
    player_values[6] = acceleration.y
    player_values[7] = acceleration.z
    player_values[8] = player.forward_speed

    # Rotation
    player_values[9] = transform.rotation.pitch
    player_values[10] = transform.rotation.roll
    player_values[11] = transform.rotation.yaw

    # Collisions
    player_values[12] = player.collision_vehicles
    player_values[13] = player.collision_pedestrians
    player_values[14] = player.collision_other

    # Intersections
    player_values[15] = 100 * player.intersection_otherlane
    player_values[16] = 100 * player.intersection_offroad

    # SUggested autopilot controler signals
    player_values[17] = control.steer
    player_values[18] = control.throttle
    player_values[19] = control.brake
    player_values[20] = control.hand_brake
    player_values[21] = control.reverse
    #player_values[frame,22] = number_of_agents
    return np.transpose(player_values)

def get_player_measurements_header():
    header = []
    header.append('platform_timestamp')
    header.append('game_timestamp')
    header.append('location_x')
    header.append('location_y')
    header.append('location_z')
    header.append('acceleration_x')
    header.append('acceleration_y')
    header.append('acceleration_z')
    header.append('forward_speed')
    header.append('pitch')
    header.append('roll')
    header.append('yaw')
    header.append('collision_vehicles')
    header.append('collision_pedestrians')
    header.append('collision_other')
    header.append('intersection_otherlane')
    header.append('intersection_offroad')
    header.append('steer')
    header.append('throttle')
    header.append('brake')
    header.append('handbrake')
    header.append('reverse')
    return DELIMITER.join(header)

def get_static_measurements_header():
    header = []
    header.append('id')
    header.append('type')
    header.append('location_x')
    header.append('location_y')
    header.append('location_z')
    header.append('orientation_x')
    header.append('orientation_y')
    header.append('yaw')
    header.append('speed_limit')
    return DELIMITER.join(header)

def get_dynamic_measurements_header(measurements):
    header = []
    for agent in measurements.non_player_agents:
        agent_type = get_agent_type(agent)
        if agent_type == 3:
            agent_id = agent.id
            agent = agent.traffic_light
            header.append(str(agent_id))
        else:
            continue

    return DELIMITER.join(header)

def get_agent_type(agent):
    if agent.vehicle.ByteSize() != 0:
        return 1 #'vehicle'
    elif agent.pedestrian.ByteSize() != 0:
        return 2 #'pedestrian'
    elif agent.traffic_light.ByteSize() != 0:
        return 3 #'traffic_light'
    elif agent.speed_limit_sign.ByteSize() != 0:
        return 4 #'speed_limit_sign'

def get_static_measurements(measurements):
    objects = {}
    for agent in measurements.non_player_agents:
        agent_id = agent.id
        agent_type = get_agent_type(agent)
        speed_limit = 0

        if agent_type == 3:
            agent = agent.traffic_light
        elif agent_type == 4:
            agent = agent.speed_limit_sign
            speed_limit = agent.speed_limit
        else:
            continue

        values = {}
        values['type'] = agent_type
        values['location_x'] = agent.transform.location.x
        values['location_y'] = agent.transform.location.y
        values['location_z'] = agent.transform.location.z
        values['orientation_x'] = agent.transform.orientation.x
        values['orientation_y'] = agent.transform.orientation.y
        values['yaw'] = agent.transform.rotation.yaw
        values['speed_limit'] = speed_limit
        objects[agent_id] = values
    return objects

def get_dynamic_measurements(measurements):
    objects = {}
    for agent in measurements.non_player_agents:
        agent_type = get_agent_type(agent)
        if agent_type == 3:
            # GREEN -> 0, YELLOW -> 1, RED -> 2
            objects[agent.id] = agent.traffic_light.state
        else:
            continue
    return objects

def save_static_measurements(static_objects, save_path):
    n_objects = len(static_objects)
    objects = np.zeros([n_objects,9])
    for i, key in enumerate(static_objects):
        values = static_objects[key]
        objects[i,0] = key
        objects[i,1] = values['type']
        objects[i,2] = values['location_x']
        objects[i,3] = values['location_y']
        objects[i,4] = values['location_z']
        objects[i,5] = values['orientation_x']
        objects[i,6] = values['orientation_y']
        objects[i,7] = values['yaw']
        objects[i,8] = values['speed_limit']

    header = get_static_measurements_header()
    np.savetxt(save_path + "/sm.csv", objects, fmt=MEASUREMENTS_PRECISION, \
        header=header, comments=COMMENTS, delimiter=DELIMITER)

def save_dynamic_measurements(header, dynamic_values, save_path):
    n_objects = len(dynamic_values)
    keys = list(dynamic_values.keys())
    n_frames = len(dynamic_values[keys[0]])
    objects = np.zeros([n_frames,n_objects])

    for i, key in enumerate(dynamic_values):
        value = dynamic_values[key] # This is a list of states
        objects[:,i] = np.asarray(value)

    np.savetxt(save_path + "/dm.csv", objects, fmt=MEASUREMENTS_PRECISION, \
        header=header, comments=COMMENTS, delimiter=DELIMITER)

def save_info(save_path, settings, args):
    now = datetime.datetime.now()
    now.strftime("%Y-%m-%d-%H-%M-%S")
    filename = save_path + 'info.txt'

    with open(filename, 'w+') as info_file:
        info = [
            'Session name: %s' % args.session_name,
            'Recording start: %s' % now.strftime("%Y-%m-%d-%H-%M-%S"),
            'Settings: %s' % args.carla_settings,
            'Frames: %i' % args.frames,
            'Autopilot: %s' % str(args.autopilot),
            'Planner: %s' % args.planner_path
            ]
        info_file.write('\n'.join(info))
