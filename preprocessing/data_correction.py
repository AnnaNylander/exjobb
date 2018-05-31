import numpy as np

def main():
    ''' This file is used for correcting the y, acceleration_y and yaw in theepisode files
    used for training the throttle, steer and brake predicting network.
    '''

    path_base = '/home/annaochjacob/Repos/exjobb/carla/PythonClient/control_module/MPC_drive/data/throttle_measurements/'

    for i in range(1,40):
        filepath = path_base + 'episode_%i.csv' %i
        episode = np.genfromtxt(filepath, delimiter=',', names=True)
        episode['location_y'] = -episode['location_y'] # reverse y axis
        episode['acceleration_y'] = -episode['acceleration_y'] # reverse y acceleration
        episode['yaw'] = -episode['yaw'] # reverse y axis
        header = ','.join(list(episode.dtype.names))
        np.savetxt(filepath, episode, comments='', delimiter=',', fmt='%.8f',
                    header=header)

if __name__ == '__main__':
    main()
