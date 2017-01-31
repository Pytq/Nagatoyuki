import time
import tensorflow as tf


sess = tf.Session()

Xp = tf.random_uniform([10, 100, 1], minval=0, maxval=1, dtype=tf.float32)
Yp = tf.random_uniform([1, 20], minval=0, maxval=1, dtype=tf.float32)

x = tf.Variable(initial_value=Xp)
y = tf.Variable(initial_value=Yp)

def batch_matmul_const2(X,Y):
    fun = lambda batch: tf.matmul(batch, Y)
    return tf.map_fn(fun, X)

def batch_matmul_const(X,Y):
    shape = X.get_shape().as_list()
    shape_y = Y.get_shape().as_list()
    if len(shape) != 3:
        raise Exception('rank must be 3')
    X = tf.reshape(X, [-1, shape[2]])
    output = tf.matmul(X, Y)
    return tf.reshape(output, [-1, shape[1], shape_y[1]])

Z1 = batch_matmul_const(x,y)
Z2 = batch_matmul_const2(x,y)
DZ1 = tf.gradients(tf.reduce_sum(Z1), y)
DZ2 = tf.gradients(tf.reduce_sum(Z2), y)
reshape = tf.reshape(x, [10,100,1,1])
expanddims = tf.expand_dims(x, axis=3)

sess.run(tf.global_variables_initializer())

for _ in range(10):
    # sess.run(tf.global_variables_initializer())
    t = time.time()
    z1 = sess.run(Z1)
    print('opt ', time.time() - t)
    # sess.run(tf.global_variables_initializer())
    t = time.time()
    z2 = sess.run(Z2)
    print('mapfn ', time.time() - t)
    t = time.time()
    sess.run(DZ1)
    print('opt diff', time.time() - t)
    t = time.time()
    sess.run(DZ2)
    print('mapfn diff', time.time() - t)
    t = time.time()
    sess.run(reshape)
    print('reshape', time.time() - t)
    t = time.time()
    sess.run(expanddims)
    print('expandfim', time.time() - t)
    print()
