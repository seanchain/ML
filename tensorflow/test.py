"""tensorflow test
"""
from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

if __name__ == "__main__":
    SESS = tf.InteractiveSession()
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    x = mnist.train.images[0]
    print (len(x))
