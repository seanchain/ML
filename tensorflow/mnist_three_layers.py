"""Mnist handwritten recognition with tensorflow and mse method
"""
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
    W1 = tf.Variable(tf.truncated_normal([784, 64], stddev=0.1))
    b1 = tf.Variable(tf.zeros([64]))
    y1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
    
    # Second hidden Layer

    W2 = tf.Variable(tf.truncated_normal([64, 32], stddev=0.1))
    b2 = tf.Variable(tf.zeros([32]))
    y2 = tf.nn.sigmoid(tf.matmul(y1, W2) + b2)

    # Outout layer
    W3 = tf.Variable(tf.truncated_normal([32, 10], stddev=0.1))
    b3 = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(y2, W3) + b3)

    # mse = tf.reduce_mean(tf.losses.mean_squared_error(labels=y_, predictions=y))
    xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.AdamOptimizer(0.003).minimize(xent)

    sess = tf.InteractiveSession()

    tf.global_variables_initializer().run()
    correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    for _ in range(20000):
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
