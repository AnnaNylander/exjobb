import numpy as np
import matplotlib.pyplot as plt

AVG = 20

def moving_average(a, n=100) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def main():
    PATH_TRAIN = '/home/annaochjacob/Repos/exjobb/nn_practice/our_network/saved/train_losses.csv'
    PATH_VALIDATE = '/home/annaochjacob/Repos/exjobb/nn_practice/our_network/saved/validation_losses.csv'
    train = np.loadtxt(PATH_TRAIN, delimiter=',')
    validate = np.loadtxt(PATH_VALIDATE, delimiter=',')

    #train_values = moving_average(train[:,1], AVG)
    #validate_values = moving_average(validate[:,1], AVG)

    train_values = train[:,1]
    validate_values = validate[:,1]

    #print(len(train[:,0]))
    #print(len(train_values))

    train_handle = plt.plot(train[:,0], train_values, label="Training loss")
    validate_handle = plt.plot(validate[:,0], validate_values, label="Validation loss")

    axes = plt.gca()
    axes.set_xlim([0,np.max(train[:,0])])
    axes.set_ylim([0,10])

    plt.show()

if __name__ == "__main__":
    main()
