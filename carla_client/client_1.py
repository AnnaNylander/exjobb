#!/usr/bin/env python3

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Basic CARLA client example."""

from __future__ import print_function

import argparse
import logging
import random
import time

from carla.client import make_carla_client
from carla.sensor import Camera
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line
import numpy as np
from preprocessing.depth_to_point_cloud import depth_to_point_cloud
from preprocessing.point_cloud_to_image import trim_to_roi

ROI = 60        # Region of interest side length, in meters.
FAR = 1000      # Far plane distance
THRESHOLD = 1.5 # Do not interpolate if difference between pixels is larger than this
INTERPOLATE = True
# Assumes subfolder 'point_cloud' and 'measurements' exist
SAVE_PATH = '/home/annaochjacob/Documents/recorded_data'
POINT_CLOUD_PRECISION = '%.3f'
MEASUREMENTS_PRECISION = '%.8f'

def run_carla_client(host, port, autopilot_on, save_images_to_disk, image_filename_format, settings_filepath):


    # We assume the CARLA server is already waiting for a client to connect at
    # host:port. To create a connection we can use the `make_carla_client`
    # context manager, it creates a CARLA client object and starts the
    # connection. It will throw an exception if something goes wrong. The
    # context manager makes sure the connection is always cleaned up on exit.
    with make_carla_client(host, port) as client:
        print('CarlaClient connected')


        if settings_filepath is None:

            # Create a CarlaSettings object. This object is a wrapper around
            # the CarlaSettings.ini file. Here we set the configuration we
            # want for the new episode.
            settings = CarlaSettings()
            settings.set(
                SynchronousMode=True,
                SendNonPlayerAgentsInfo=True,
                NumberOfVehicles=20,
                NumberOfPedestrians=40,
                WeatherId=random.choice([1, 3, 7, 8, 14]))
            settings.randomize_seeds()

            # Now we want to add a couple of cameras to the player vehicle.
            # We will collect the images produced by these cameras every
            # frame.

            # The default camera captures RGB images of the scene.
            camera0 = Camera('CameraRGB')
            # Set image resolution in pixels.
            camera0.set_image_size(800, 600)
            # Set its position relative to the car in centimeters.
            camera0.set_position(30, 0, 130)
            settings.add_sensor(camera0)

            # Let's add another camera producing ground-truth depth.
            camera1 = Camera('CameraDepth', PostProcessing='Depth')
            camera1.set_image_size(800, 600)
            camera1.set_position(30, 0, 130)
            settings.add_sensor(camera1)

        else:

            # Alternatively, we can load these settings from a file.
            with open(settings_filepath, 'r') as fp:
                settings = fp.read()

        # Now we load these settings into the server. The server replies
        # with a scene description containing the available start spots for
        # the player. Here we can provide a CarlaSettings object or a
        # CarlaSettings.ini file as string.
        scene = client.load_settings(settings)

        # Choose one player start at random.
        number_of_player_starts = len(scene.player_start_spots)
        player_start = random.randint(0, max(0, number_of_player_starts - 1))

        # Notify the server that we want to start the episode at the
        # player_start index. This function blocks until the server is ready
        # to start the episode.
        print('Starting new episode...')
        client.start_episode(player_start)

        # Iterate every frame in the episode.
        n_frames = 3
        measurement_values = np.zeros([n_frames + 1,22])
        for frame in range(1,n_frames):

            # Read the data produced by the server this frame.
            measurements, sensor_data = client.read_data()

            print(measurements)

            # Print some of the measurements.
            print_measurements(measurements)

            # Save the images to disk if requested.
            if save_images_to_disk:
                for name, image in sensor_data.items():
                    image.save_to_disk(image_filename_format.format(1, name, frame))

                    head = sensor_data['CameraDepthHead'].data
                    tail = sensor_data['CameraDepthTail'].data
                    left = sensor_data['CameraDepthLeft'].data
                    right = sensor_data['CameraDepthRight'].data

                    # Convert depth maps to 3D point cloud
                    point_cloud = depth_to_point_cloud(head, tail, left, right, FAR, interpolate=INTERPOLATE, threshold=THRESHOLD)
                    # Trim point cloud to only contain points within the region of interest
                    point_cloud = trim_to_roi(point_cloud,ROI)
                    # Reduce
                    # Save point cloud for this frame
                    np.savetxt(SAVE_PATH + "/point_cloud/pc_%i.csv" %frame, point_cloud,fmt=POINT_CLOUD_PRECISION)

                    player = measurements.player_measurements
                    control = player.autopilot_control
                    transform = player.transform
                    acceleration = player.acceleration

                    measurement_values[frame,0] = measurements.platform_timestamp
                    measurement_values[frame,1] = measurements.game_timestamp

                    # Location
                    measurement_values[frame,2] = transform.location.x / 100 #  (cm -> m )
                    measurement_values[frame,3] = transform.location.y / 100
                    measurement_values[frame,4] = transform.location.z / 100

                    # Acceleration and forward speed
                    measurement_values[frame,5] = acceleration.x
                    measurement_values[frame,6] = acceleration.y
                    measurement_values[frame,7] = acceleration.z
                    measurement_values[frame,8] = player.forward_speed

                    # Rotation
                    measurement_values[frame,9] = transform.rotation.pitch
                    measurement_values[frame,10] = transform.rotation.roll
                    measurement_values[frame,11] = transform.rotation.yaw

                    # Collisions
                    measurement_values[frame,12] = player.collision_vehicles
                    measurement_values[frame,13] = player.collision_pedestrians
                    measurement_values[frame,14] = player.collision_other

                    # Intersections
                    measurement_values[frame,15] = 100 * player.intersection_otherlane
                    measurement_values[frame,16] = 100 * player.intersection_offroad

                    # SUggested autopilot controler signals
                    measurement_values[frame,17] = control.steer
                    measurement_values[frame,18] = control.throttle
                    measurement_values[frame,19] = control.brake
                    measurement_values[frame,20] = control.hand_brake
                    measurement_values[frame,21] = control.reverse
                    #measurement_values[frame,22] = number_of_agents

                    #save_driving_data(measurements, sensor_data, frame)

            # Now we have to send the instructions to control the vehicle.
            # If we are in synchronous mode the server will pause the
            # simulation until we send this control.

            if not autopilot_on:

                client.send_control(
                    steer=random.uniform(-0.1, 0.1),
                    throttle=0.5,
                    brake=0.0,
                    hand_brake=False,
                    reverse=False)

            else:

                # Together with the measurements, the server has sent the
                # control that the in-game autopilot would do this frame. We
                # can enable autopilot by sending back this control to the
                # server. We can modify it if wanted, here for instance we
                # will add some noise to the steer.
                control = measurements.player_measurements.autopilot_control
                control.steer += random.uniform(-0.1, 0.1)
                client.send_control(control)

        # Save measurements of whole episode to one file
        np.savetxt(SAVE_PATH + "/measurements/m.csv", measurement_values, fmt = MEASUREMENTS_PRECISION)


def print_measurements(measurements):
    number_of_agents = len(measurements.non_player_agents)
    player_measurements = measurements.player_measurements
    message = 'Vehicle at ({pos_x:.1f}, {pos_y:.1f}), '
    message += '{speed:.2f} km/h, '
    message += 'Collision: {{vehicles={col_cars:.0f}, pedestrians={col_ped:.0f}, other={col_other:.0f}}}, '
    message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road, '
    message += '({agents_num:d} non-player agents in the scene)'
    message = message.format(
        pos_x=player_measurements.transform.location.x / 100, # cm -> m
        pos_y=player_measurements.transform.location.y / 100,
        speed=player_measurements.forward_speed,
        col_cars=player_measurements.collision_vehicles,
        col_ped=player_measurements.collision_pedestrians,
        col_other=player_measurements.collision_other,
        other_lane=100 * player_measurements.intersection_otherlane,
        offroad=100 * player_measurements.intersection_offroad,
        agents_num=number_of_agents)
    print_over_same_line(message)


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '-i', '--images-to-disk',
        action='store_true',
        help='save images to disk')
    argparser.add_argument(
        '-c', '--carla-settings',
        metavar='PATH',
        default=None,
        help='Path to a "CarlaSettings.ini" file')

    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    while True:
        try:

            run_carla_client(
                host=args.host,
                port=args.port,
                autopilot_on=args.autopilot,
                save_images_to_disk=args.images_to_disk,
                image_filename_format='_images/episode_{:0>3d}/{:s}/image_{:0>5d}.png',
                settings_filepath=args.carla_settings)

            print('Done.')
            return

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
