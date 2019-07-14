import tensorflow as tf
import numpy as np

xs = np.random.randint(46,99,100)
ys = 1.7*xs

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
w = tf.Variable(0.1)
b = tf.Variable(0.1)

y_ = tf.multiply(w,x)+b
cost = tf.reduce_sum(tf.pow((y-y_),2))
train_step = tf.train.GradientDescentOptimizer(0.02).minimize(cost)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for _ in range(10):
    sess.run(train_step,feed_dict={x:xs,y:ys})
    print(w.eval(sess))
    print(b.eval(sess))
    print("##")
