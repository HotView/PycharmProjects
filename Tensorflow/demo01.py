import tensorflow as tf
import numpy as np

x_data = np.random.rand(100).astype(np.float32)
y_data  =x_data*0.1+0.3

# create tensorflow structure start
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))
y = Weights*x_data+biases
loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init  = tf.initialize_all_variables()
# create tensorflow structure start
session = tf.Session()
session.run(init) # 激活神经网络

for step in range(201):
    session.run(train)
    if step%20==0:
        print(step,session.run(Weights),session.run(biases))

