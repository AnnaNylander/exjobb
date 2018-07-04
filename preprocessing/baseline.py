import numpy as np
import os
import util

def main():
    ''' Returns the MSE of the specified dataset by comparing the ground truth
    trajectories with a trajectory straight ahead at the current speed.'''

    dataset_path = '/media/annaochjacob/crucial/dataset/flexible/validate/'
    time_steps = np.array(range(1,31)) # We use all time steps from 1 to 30 inclusive.

    all_errors = []

    # Go through all folders in the dataset, e.g. "apple" or "cat"
    for folder in os.listdir(dataset_path):
        folder_errors = []
        # Load values file with coordinates
        values_path = dataset_path + folder + '/values.csv'
        values = np.loadtxt(values_path , delimiter=',', skiprows=1)

        for i, row in enumerate(values):
            # Simply skip the last time steps
            if i + max(time_steps) >= len(values):
                continue

            # Calculate the ground truth relative coordinates
            x, y, yaw, speed = row[[0,1,2,4]]
            w_coord = values[i + time_steps]
            w_coord = w_coord[:,[0,1]].transpose()
            r_coord = util.world_to_relative(x, y, yaw, w_coord)

            # Calculate baseline trajectory
            b_coord = get_baseline_trajectory(speed, time_steps)

            # Calculate error between baseline trajectory and ground truth
            error = mse(b_coord, r_coord)
            folder_errors.append(error)
            all_errors.append(error)

        folder_mean = np.mean(np.array(folder_errors))
        print('MSE for',folder,':',folder_mean)

    all_mean = np.mean(np.array(all_errors))
    print('MSE ON THE WHOLE DATASET:', all_mean)


def get_baseline_trajectory(forward_speed, time_steps):
    ''' Returns a baseline trajectory which is formed by continuing in the
    current direction with the current forward speed (m/s).

    The input "time_spets" is a list which enumerates the time steps to include.
    Each time step is assumed to be 0.1s'''

    x = np.zeros(max(time_steps))
    y = np.arange(1, max(time_steps) + 1) * forward_speed * 0.1
    return np.array([x,y])

def mse(x,y):
    ''' Returns the mean squared error between x and y'''
    return ((x-y)**2).mean(axis=None)

if __name__ == '__main__':
    main()
