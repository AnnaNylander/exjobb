import numpy as np
import matplotlib.pyplot as plt

PATH_TRAIN = '/home/annaochjacob/Repos/exjobb/nn_practice/our_network/saved/train_losses.csv'
PATH_VALIDATE = '/home/annaochjacob/Repos/exjobb/nn_practice/our_network/saved/validation_losses.csv'
train = np.loadtxt(PATH_TRAIN, delimiter=',')
validate = np.loadtxt(PATH_VALIDATE, delimiter=',')

train_handle = plt.plot(train[:,0], train[:,1], label="Training loss")
validate_handle = plt.plot(validate[:,0], validate[:,1], label="Validation loss")

axes = plt.gca()
axes.set_xlim([0,np.max(train[:,0])])
axes.set_ylim([0,1000])

plt.show()
