"""
A simple example of mnist handwritten recognition
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

def main():
    """
    main function
    """
    start = time.time()
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # print mnist.train.images[0]
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b

    y_ = tf.placeholder(tf.float32, [None, 10])

    xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(xent)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    print(sess.run(accuracy, feed_dict={x:mnist.test.images,
                                            y_: mnist.test.labels}))
                        
    end = time.time()
    print ("Time elapsed " + str(end - start))

if __name__ == '__main__':
    main()
