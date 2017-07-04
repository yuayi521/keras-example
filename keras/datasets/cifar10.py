from __future__ import absolute_import
from keras.datasets.cifar import load_batch
from keras import backend as K
import numpy as np
import os


def load_data():
    """
    Loads CIFAR10 dataset.
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    path = '/home/yuquanjie/Download/cifar-10-batches-py'
    num_train_samples = 50000
    # good habit, define numpy array and add type of element
    x_train = np.zeros((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.zeros((num_train_samples,), dtype='uint8')
    # read data from 6 files, data_batch_1 , data_batch_2, ...., data_batch_6
    # similar to my deep direct task, all training data sotred in one file that is to large, so use 6 files to
    # store training data
    for i in range(1, 6):
        # I'm used to utilizing bytes(i)
        fpath = os.path.join(path, 'data_batch_' + str(i))
        data, labels = load_batch(fpath)
        # should learn numpy arry operaions
        x_train[(i - 1) * 10000: i * 10000, :, :, :] = data
        y_train[(i - 1) * 10000: i * 10000] = labels
    print 'x_train.shape is {0}'.format(x_train.shape)
    print 'y_train.shape is {0}'.format(y_train.shape)
    # load test data, test data is not large, so use 1 file to store test data
    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)
    # gives a new shape to an array without changing its data
    y_train = np.reshape(y_train, (len(y_train), 1))
    print 'y_train.shape is {0}'.format(y_train.shape)
    y_test = np.reshape(y_test, (len(y_test), 1))

    # channel is the last, for example in direct regression, iamge is (320, 320, 3)
    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)

if __name__ == '__main__':
    load_data()
