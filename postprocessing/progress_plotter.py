import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Plot training progress')
parser.add_argument('--path', metavar='PATH', type=str,
                    dest='path', default='/saved/',
                    help='Path to folder with saved losses')
parser.add_argument('--avg', metavar='N', type=int,
                    dest='avg', default=20,
                    help='Moving average windows size')

parser.add_argument('--ylim', metavar='N', type=int,
			dest='ylim', default=10, help='Y-axis positive limit')

args = parser.parse_args()

def moving_average(a, n=100) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def main():
    PATH_TRAIN = args.path + 'train_losses.csv'
    PATH_VALIDATE = args.path + 'validation_losses.csv'
    train = np.loadtxt(PATH_TRAIN, delimiter=',')
    validate = np.loadtxt(PATH_VALIDATE, delimiter=',')

    #train_values = moving_average(train[:,1], args.avg)
    #validate_values = moving_average(validate[:,1], args.avg)

    train_values = train[:,1]
    validate_values = validate[:,1]

    #print(len(train[:,0]))
    #print(len(train_values))

    train_handle = plt.plot(train[:,0], train_values, label="Training loss")
    validate_handle = plt.plot(validate[:,0], validate_values, label="Validation loss")

    axes = plt.gca()
    axes.set_xlim([0,np.max(train[:,0])])
    axes.set_ylim([0,args.ylim])

    plt.show()

if __name__ == "__main__":
    main()
