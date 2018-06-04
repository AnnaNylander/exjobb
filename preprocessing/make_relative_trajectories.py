import numpy as np
import util
import os

def main():
    save_path = '/home/annaochjacob/Repos/exjobb/preprocessing/path_clustering/'
    base_path = '/media/annaochjacob/crucial/dataset/flexible/train/'
    future_idxs = list(range(1,31)) # Set this to whatever future steps to use
    folders = os.listdir(base_path)

    for folder in folders:
        print('Making trajectories for %s' % folder)
        values_path = base_path + folder + '/values.csv'
        trajectories = get_trajectories(values_path, future_idxs)

        # Save trajectories for the current values file
        np.savetxt(base_path + folder + '/trajectories.csv', trajectories,
                    comments='', delimiter=',', fmt='%.8f')

def get_trajectories(values_path, future_idxs):
    ''' Returns a matrix where each row is the relative trajectory.'''
    values = temp = np.genfromtxt(values_path, delimiter=',',skip_header=True)
    n_relevant_points = len(values) - len(future_idxs)
    trajectories = np.zeros([n_relevant_points,len(future_idxs)*2])

    for i in range(0, n_relevant_points):
        r_coord = get_output(values, i, future_idxs)
        trajectories[i] = r_coord.reshape(-1)

    return trajectories

def get_output(values, current_idx, future_idxs):
    ''' Returns the relative coordinates at time steps future_idxs
        from current_idx as one row.'''
    current_frame = values[current_idx]
    x, y, yaw = current_frame[[0,1,2]]
    w_coord = values[current_idx + np.array(future_idxs), :]
    w_coord = w_coord[:, [0,1]] # Keep only x and y
    w_coord = w_coord.transpose()
    r_coord = util.world_to_relative(x, y, yaw, w_coord)

    return r_coord.reshape(-1,len(future_idxs)*2)

if __name__ == '__main__':
    main()
