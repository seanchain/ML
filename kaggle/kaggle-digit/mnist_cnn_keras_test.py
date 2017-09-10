'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import numpy as np
import keras
from keras.models import load_model


def evaluate():
    """
    Train and Evaluation Function
    """

    # the data, shuffled and split between train and test sets
    x_test = np.load('x_test.npy')
    x_test = x_test.reshape(28000, 28, 28, 1)
    x_test = x_test.astype('float32')
    x_test /= 255
    model = load_model('./cnn-ckpt-56.hdf5')
    outputs = model.predict(x_test)
    indexes = []
    for output in outputs:
        index = max(enumerate(output),key=lambda x: x[1])[0]
        indexes.append(index)
    start = 1
    with open('result.csv', 'wb') as f:
        f.write(b'ImageId,Label\n')
        for index in indexes:
            f.write(str(start) + ',' + str(index) + '\n')
            start += 1
def main():
    """
    Main function
    """
    evaluate()

if __name__ == '__main__':
    main()
