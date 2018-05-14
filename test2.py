import tensorflow as tf

a1 = tf.placeholder(tf.int16)
a2 = tf.placeholder(tf.int16)
x1 = tf.constant([2, 3, 4])
x2 = tf.constant([4, 0, 1])
b = tf.add(x1, x2)

li1 = [2, 3, 4]
li2 = [4, 0, 1]

with tf.Session() as sess:
    print (sess.run(b, feed_dict={a1: li1, a2: li2}))
