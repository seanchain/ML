'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import load_model
import img2vec


def evaluate():
    """
    Train and Evaluation Function
    """

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_test = x_test.reshape(10000, 28, 28, 1)
    x_test = x_test.astype('float32')
    x_test /= 255
    y_test = keras.utils.to_categorical(y_test, 10)
    model = load_model('cnn-ckpt-10.hdf5')
    new_input = img2vec.getTests()
    new_input /= 255
    new_input = new_input.reshape(6, 28, 28, 1)
    outputs = model.predict(new_input)
    print(outputs)
    for output in outputs:
        index = max(enumerate(output),key=lambda x: x[1])[0]
        print(index)
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

def main():
    """
    Main function
    """
    evaluate()

if __name__ == '__main__':
    main()
