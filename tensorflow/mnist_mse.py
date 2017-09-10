"""Mnist handwritten rec/gnition with tensorflow and mse method
"""
from __future__ import print_function
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def main():
    """Main function here
    """
    start = time.time()
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # Input layer
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])

    x_image = tf.reshape(x, [-1, 784])


    # First hidden layer
    W1 = tf.Variable(tf.truncated_normal([784, 30], stddev=0.1))
    b1 = tf.Variable(tf.zeros([30]))
    y1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)

    # Outout layer
    W2 = tf.Variable(tf.truncated_normal([30, 10], stddev=0.1))
    b2 = tf.Variable(tf.zeros([10]))
    y = tf.nn.sigmoid(tf.matmul(y1, W2) + b2)

    mse = tf.reduce_mean(tf.losses.mean_squared_error(labels=y_, predictions=y))
    train_step = tf.train.AdamOptimizer(0.003).minimize(mse)

    sess = tf.InteractiveSession()

    tf.global_variables_initializer().run()
    correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    for _ in range(7200):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        if _ % 500 == 0:
            print("Iter #%i - %g" %(_ ,sess.run(acc, feed_dict={x:batch_xs, y_:batch_ys})))

    print(sess.run(acc, feed_dict={x: mnist.test.images,
                             y_: mnist.test.labels}))
    saver = tf.train.Saver()
    saver.save(sess, 'my_test_model')
    end = time.time()

    print ("Time elapsed %g seconds" % (end - start))


if __name__ == "__main__":
    main()
