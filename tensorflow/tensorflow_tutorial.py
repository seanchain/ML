"""
This file contains the official tutorial codes provided
by the tensorflow website
"""

import tensorflow as tf

def main():
    """
    Main function
    """
    # Tensorflow core programs consists two sections
    # 1. build the computational graph
    # 2. run the computational graph
    node1 = tf.constant(3.0, dtype=tf.float32)
    node2 = tf.constant(4.0) # also tf.float32 implicitly
    print node1, node2 # node1 and 2 are two floating point tensors

    # We must run the computational graph with a session
    sess = tf.Session()
    print sess.run([node1, node2])

    node3 = tf.add(node1, node2)
    print "Node3: ", node3
    print "Sess.run(node3)", sess.run(node3)
    # A graph can be parameterized to accept external inputs
    # which is placeholders

    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    adder_node = a + b # + provides a shortcut for tf.add(a, b)
    print sess.run(adder_node, {a: 3, b: 5})
    print sess.run(adder_node, {a: [1, 3], b: [2, 4]})

    add_and_triple = adder_node * 3
    print sess.run(add_and_triple, {a: 3, b: 4.5})

    # Variables allow us to add trainable parameters to a graph
    W = tf.Variable([.3], dtype=tf.float32)
    b = tf.Variable([-.3], dtype=tf.float32)
    x = tf.placeholder(tf.float32)
    linear_model = W * x + b

    # contants are initialized with tf.constant
    # to initialize all the variables use
    init = tf.global_variables_initializer()
    sess.run(init)
    print sess.run(linear_model, {x:[1, 2, 3, 4]})

    # we need a y placeholder to evaluate the model of training data
    y = tf.placeholder(tf.float32)
    squared_detas = tf.square(linear_model - y)
    loss = tf.reduce_sum(squared_detas)
    print sess.run(loss, {x: [1, 2, 3, 4], y:[0, -1, -2, -3]}) # the loss

    # we can improve by manually reassigning the values of W and b
    fixW = tf.assign(W, [-1.])
    fixb = tf.assign(b, [1.])
    sess.run([fixW, fixb])
    print sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

if __name__ == '__main__':
    main()
