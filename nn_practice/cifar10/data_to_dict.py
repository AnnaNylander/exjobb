def getTrainingSet():
    import numpy
    dic1 = getTrainingBatch(1)
    dic2 = getTrainingBatch(2)
    dic3 = getTrainingBatch(3)
    dic4 = getTrainingBatch(4)
    dic5 = getTrainingBatch(5)
    bl = b'batch_label'
    l = b'labels'
    d = b'data'
    f = b'filenames'
    data = numpy.concatenate((dic1.get(d),dic2.get(d),dic3.get(d),dic4.get(d),dic5.get(d)))
    dic = {bl:dic1.get(bl)+dic2.get(bl)+dic3.get(bl)+dic4.get(bl)+dic5.get(bl),
        l:dic1.get(l)+dic2.get(l)+dic3.get(l)+dic4.get(l)+dic5.get(l),
        d:data,
        f:dic1.get(f)+dic2.get(f)+dic3.get(f)+dic4.get(f)+dic5.get(f)}
    return dic

def getTrainingBatch(batch):
    """returns specified batch (int between 1 and 5) as a dict"""
    return unpickle('data/cifar-10-batches-py/data_batch_'+str(batch))

def getTestSet():
    """return test batch as a dict"""
    return unpickle('data/cifar-10-batches-py/test_batch')

def unpickle(file):
    """ Will return a dict with the following elements:
    data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a
        32x32 colour image. The first 1024 entries contain the red channel values,
        the next 1024 the green, and the final 1024 the blue. The image is stored
        in row-major order, so that the first 32 entries of the array are the red
        channel values of the first row of the image.
    labels -- a list of 10000 numbers in the range 0-9. The number at index i
        indicates the label of the ith image in the array data."""
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
