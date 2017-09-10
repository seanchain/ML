"""
This file tells how tensorflow trains networks
"""

import tensorflow as tf

def main():
    """
    Main function of the trainnig process
    """
    W = tf.Variable([.3], dtype=tf.float32)
    b = tf.Variable([-.3], dtype=tf.float32)
    x = tf.placeholder(tf.float32)
    linear_model = W * x + b

    sess = tf.Session()

    init = tf.global_variables_initializer()
    sess.run(init)

    # we need a y placeholder to evaluate the model of training data
    y = tf.placeholder(tf.float32)
    squared_detas = tf.square(linear_model - y)
    loss = tf.reduce_sum(squared_detas)
    optimizer = tf.train.GradientDescentOptimizer(0.02)
    train = optimizer.minimize(loss)

    sess.run(init) # reset all the values to incorrect defaults
    for i in range(1000):
        sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})
    
    print sess.run([W, b])


if __name__ == '__main__':
    main()