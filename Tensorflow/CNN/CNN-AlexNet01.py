import Tensorflow.CNN.create_and_read_TFRecord2 as read2
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

X_train,y_train = read2.get_file("E:/cat_and_dog/train2")
# print("X_train lens:",len(X_train))
# print("y_train lens:",len(y_train))
image_batch,label_batch = read2.get_batch(X_train,y_train,227,227,200,2048)
def batch_norm(inputs,is_training,is_conv_out= True,decay=0.999):
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]),trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]),trainable=False)
    if is_training:
        if is_conv_out:
            batch_mean,batch_var = tf.nn.moments(inputs,[0,1,2])
        else:
            batch_mean, batch_var = tf.nn.moments(inputs, [0])
        train_mean = tf.assign(pop_mean,pop_mean*decay+batch_mean*(1-decay))
        train_var = tf.assign(pop_var,pop_var*decay+batch_var*(1-decay))
        with tf.control_dependencies([train_mean,train_var]):
            return tf.nn.batch_normalization(inputs,batch_mean,batch_var,beta,scale,0.001)
    else:
        return tf.nn.batch_normalization(inputs,pop_mean,pop_var,beta,scale,0.001)
learning_rate = 1e-4
training_iters = 200
batch_size = 200
display_step = 5
n_classes = 2
n_fc1 = 4096
n_fc2 = 2048
x = tf.placeholder(tf.float32,[None,227,227,3])
y = tf.placeholder(tf.int32,[None,n_classes])
W_conv = {
    'conv1':tf.Variable(tf.truncated_normal([11,11,3,96],stddev=0.0001)),
    'conv2':tf.Variable(tf.truncated_normal([5,5,96,256],stddev=0.01)),
    'conv3':tf.Variable(tf.truncated_normal([3,3,256,384],stddev=0.01)),
    'conv4':tf.Variable(tf.truncated_normal([3,3,384,384],stddev=0.01)),
    'conv5':tf.Variable(tf.truncated_normal([3,3,384,256],stddev=0.01)),
    'fc1':tf.Variable(tf.truncated_normal([6*6*256,n_fc1],stddev=0.1)),# 6*6 表示最后生成的图像的大小
    'fc2':tf.Variable(tf.truncated_normal([n_fc1,n_fc2],stddev=0.1)),
    'fc3':tf.Variable(tf.truncated_normal([n_fc2,n_classes],stddev=0.1))
}
b_conv = {
    'conv1':tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[96])),
    'conv2':tf.Variable(tf.constant(0.1,dtype=tf.float32,shape=[256])),
    'conv3':tf.Variable(tf.constant(0.1,dtype=tf.float32,shape=[384])),
    'conv4':tf.Variable(tf.constant(0.1,dtype=tf.float32,shape=[384])),
    'conv5':tf.Variable(tf.constant(0.1,dtype=tf.float32,shape=[256])),
    'fc1':tf.Variable(tf.constant(0.1,dtype=tf.float32,shape=[n_fc1])),
    'fc2':tf.Variable(tf.constant(0.1,dtype=tf.float32,shape=[n_fc2])),
    'fc3':tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[n_classes]))
}
x_image = tf.reshape(x,[-1,227,227,3])
# 卷积层1
conv1 = tf.nn.conv2d(x_image,W_conv['conv1'],strides=[1,4,4,1],padding='VALID')
conv1 = tf.nn.bias_add(conv1,b_conv['conv1'])
conv1 = batch_norm(conv1,True)
conv1 = tf.nn.relu(conv1)
pool1 = tf.nn.avg_pool(conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')# 池化层1
norm1 = tf.nn.lrn(pool1,5,bias=1.0,alpha=0.001/9.0,beta=0.75)#LRN层
# 卷积层2
conv2 = tf.nn.conv2d(norm1,W_conv['conv2'],strides=[1,1,1,1],padding='SAME')
conv2 =tf.nn.bias_add(conv2,b_conv['conv2'])
conv2 = batch_norm(conv2,True)
conv2 = tf.nn.relu(conv2)
pool2 = tf.nn.avg_pool(conv2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')
norm2 = tf.nn.lrn(pool2,5,bias=1.0,alpha=0.001/9.0,beta=0.75)
# 卷积层3
conv3 = tf.nn.conv2d(norm2,W_conv['conv3'],strides=[1,1,1,1],padding='SAME')
conv3 =tf.nn.bias_add(conv3,b_conv['conv3'])
conv3 = batch_norm(conv3,True)
conv3 = tf.nn.relu(conv3)
# 卷积层4
conv4 = tf.nn.conv2d(conv3,W_conv['conv4'],strides=[1,1,1,1],padding='SAME')
conv4 =tf.nn.bias_add(conv4,b_conv['conv4'])
conv4 = batch_norm(conv4,True)
conv4 = tf.nn.relu(conv4)
# 卷积层5
conv5 = tf.nn.conv2d(conv4,W_conv['conv5'],strides=[1,1,1,1],padding='SAME')
conv5 =tf.nn.bias_add(conv5,b_conv['conv5'])
conv5 = batch_norm(conv5,True)
conv5 = tf.nn.relu(conv5)
pool5 = tf.nn.avg_pool(conv5,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')
print(pool5.shape)
reshape = tf.reshape(pool5,[-1,6*6*256])
#全连接1层
fc1 = tf.add(tf.matmul(reshape,W_conv['fc1']),b_conv['fc1'])
fc1 = batch_norm(fc1,True,False)
fc1 = tf.nn.relu(fc1)
fc1 = tf.nn.dropout(fc1,0.5)
#全连接2层
fc2 = tf.add(tf.matmul(fc1,W_conv['fc2']),b_conv['fc2'])
fc2 =batch_norm(fc2,True,False)
fc2 = tf.nn.relu(fc2)
fc2 = tf.nn.dropout(fc2,0.5)
#全连接3层
fc3  =tf.add(tf.matmul(fc2,W_conv['fc3']),b_conv['fc3'])
# 定义损失函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc3,labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
# 评估模型
correct_pred = tf.equal(tf.argmax(fc3,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
init = tf.global_variables_initializer()
def onehot(labels):
    n_sample = len(labels)
    n_class = max(labels)+1
    onehot_labels = np.zeros((n_sample,n_class))
    onehot_labels[np.arange(n_sample),labels] = 1
    return onehot_labels
save_model = "E:/cat_and_dog/model/"
def train(opech):
    with tf.Session() as sess:
        sess.run(init)

        train_writer = tf.summary.FileWriter("E:/cat_and_dog/log",sess.graph)
        saver = tf.train.Saver()

        c = []
        start_time = time.time()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        step = 0
        for i in range(opech):
            step = i
            image,label = sess.run([image_batch,label_batch])

            labels = onehot(label)

            sess.run(optimizer,feed_dict={x:image,y:labels})
            loss_record = sess.run(loss,feed_dict={x:image,y:labels})
            print("now the loss is %f "%loss_record)

            c.append(loss_record)
            end_time = time.time()
            print("time:",(end_time-start_time))
            start_time = end_time
            print("------------%d onpech is finished------------"%i)
        print("Optimization Finished!")
        saver.save(sess,save_model)
        print("Model Save Finished!")

        coord.request_stop()
        coord.join(threads)
        plt.plot(c)
        plt.xlabel("Iter")
        plt.ylabel("loss")
        plt.title("lr= %f,ti =%d,bs=%d"%(learning_rate,training_iters,batch_size))
        plt.tight_layout()
        plt.savefig("E:/cat_and_dog/fig/cat_and_dog_AlexNet.jpg",dpi= 200)
from PIL import Image

def per_class(imagefile):
    image = Image.open(imagefile)
    image = image.resize([227,227])
    image_array = np.array(image)

    image = tf.cast(image_array,tf.float32)
    image = tf.image.per_image_standardization(image)
    imags = tf.reshape(image,[227,227,3])

    saver = tf.train.Saver()
    with tf.Session() as sess:
        save_model = tf.train.latest_checkpoint("E:/cat_and_dog/VGGmodel/")
        saver.restore(sess,save_model)
        image = tf.reshape(image,[1,227,227,3])
        image = sess.run(image)
        prediction = sess.run(fc3,feed_dict={x:image})

        max_index = np.argmax(prediction)
        if max_index==0:
            return "cat"
        else:
            return "dog"

#train(5000)
for i in range(1,20):
    imagefile = "E:/cat_and_dog/test1/"+str(i)+".jpg"
    print(imagefile)
    print(per_class(imagefile))