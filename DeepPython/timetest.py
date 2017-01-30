import time
import tensorflow as tf


sess = tf.Session()

Xp = tf.random_uniform([10000000, 3], minval=0, maxval=1, dtype=tf.float32)
Yp = tf.random_uniform([10000000, 3], minval=0, maxval=1, dtype=tf.float32)

X = tf.Variable(initial_value=Xp)
Y = tf.Variable(initial_value=Yp)
sess.run(tf.global_variables_initializer())

Z = X*Y
Zp = tf.map_fn(lambda line: line*Y, X)

for _ in range(10):
    t = time.time()
    z = sess.run(Z)
    print(time.time() - t)
    t = time.time()
    zp = sess.run(Z)
    print(time.time() - t)
    print()