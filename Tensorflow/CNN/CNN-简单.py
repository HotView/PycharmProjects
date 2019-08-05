import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
sess = tf.Session()
#------------------------------------------------------
# 加载数据集，将原图像784*1的形状转化为28*28的数组
#------------------------------------------------------
data_dir = '../data'
mnist = read_data_sets(data_dir)
train_xdata = np.array([np.reshape(x,(28,28)) for x in mnist.train.images])
test_xdata = np.array([np.reshape(x,(28,28))for x in mnist.test.images])
train_labels = mnist.train.labels
test_labels =mnist.test.labels
print(train_xdata.shape)
#------------------------------------------------------
# 设置模型参数，由于图像是灰度图，所以该图像深度为1，即颜色通道为1,
#------------------------------------------------------
batch_size = 100
learning_rate = 0.005
evalution_size = 500
image_width = train_xdata[0].shape[0]
image_height = train_xdata[0].shape[1]
target_size = max(train_labels)+1
num_channels = 1
generations = 500
eval_every = 5
conv1_features = 25
conv2_features = 50
max_pool_size1 = 2
max_pool_size2 = 2
fully_connected_size1 = 100
#------------------------------------------------------
# 为数据集声明占位符。同时，声明训练数据集变量和测试集变量
#------------------------------------------------------
x_input_shape = (batch_size,image_width,image_height,num_channels)
x_input = tf.placeholder(tf.float32,shape=x_input_shape)
y_target = tf.placeholder(tf.int32,shape=(batch_size))
eval_input_shape = (evalution_size,image_width,image_height,num_channels)
eval_input = tf.placeholder(tf.float32,shape=eval_input_shape)
eval_target = tf.placeholder(tf.int32,shape=(evalution_size))
#------------------------------------------------------
# 声明卷积层的权重和偏置，权重和偏置的参数在前面已设置过
#------------------------------------------------------
conv1_weight = tf.Variable(tf.truncated_normal([4,4,num_channels,conv1_features],stddev=0.1,dtype=tf.float32))
conv1_bias = tf.Variable(tf.zeros([conv1_features],dtype=tf.float32))
conv2_weight = tf.Variable(tf.truncated_normal([4,4,num_channels,conv2_features],stddev=0.1,dtype=tf.float32))
conv2_bias = tf.Variable(tf.zeros([conv2_features],dtype=tf.float32))
#------------------------------------------------------
# 声明全连接层的权重和偏置
#------------------------------------------------------
resulting_width = image_width//(max_pool_size1*max_pool_size2)
resulting_height = image_height//(max_pool_size1*max_pool_size2)
full1_input_size = resulting_width*resulting_height*conv2_features
full1_weight = tf.Variable(tf.truncated_normal([full1_input_size,fully_connected_size1],stddev=0.1,dtype=tf.float32))
full1_bias = tf.Variable(tf.truncated_normal([fully_connected_size1],stddev=0.1,dtype=tf.float32))
full2_weight = tf.Variable(tf.truncated_normal([fully_connected_size1,target_size],stddev=0.1,dtype=tf.float32))
full2_bias = tf.Variable(tf.truncated_normal([target_size],stddev=0.1,dtype=tf.float32))
#------------------------------------------------------
# 声明算法模型。首先，创建一个模型函数my_conv_net()，注意该函数的层权重和偏置
#------------------------------------------------------
def my_conv_net(input_data):
    # 第一层的卷积-激活-池化层
    conv1 = tf.nn.conv2d(input_data,conv1_weight,strides=[1,1,1,1],padding='SAME')
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_bias))
    max_pool1 = tf.nn.max_pool(relu1,ksize=[1,max_pool_size1,max_pool_size1,1],strides=[1,max_pool_size1,max_pool_size1,1],padding='SAME')
    # 第二层的卷积-激活-池化层
    conv2 = tf.nn.conv2d(max_pool1,conv2_weight,strides=[1,1,1,1],padding='SAME')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_bias))
    max_pool2 = tf.nn.max_pool(relu2,ksize=[1,max_pool_size2,max_pool_size2,1],strides=[1,max_pool_size2,max_pool_size2,1],padding='SAME')
    # 改变输出格式为1*N layer for next fully connected layer
    final_conv_shape = max_pool2.get_shape().as_list()
    final_shape = final_conv_shape[1]*final_conv_shape[2]*final_conv_shape[3]
    flat_output = tf.reshape(max_pool2,[final_conv_shape[0],final_shape])
    # 第一层的全连接层
    fully_connected1 = tf.nn.relu(tf.add(tf.matmul(flat_output,full1_weight),full1_bias))
    # 第二层的全连接层
    final_model_output = tf.add(tf.matmul(fully_connected1,full2_weight),full2_bias)
    return final_model_output
#------------------------------------------------------
# 声明模型训练
#------------------------------------------------------
model_output = my_conv_net(x_input)
test_model_output = my_conv_net(eval_input)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model_output,labels=y_target))
prediction = tf.nn.softmax(model_output)
test_prediction = tf.nn.softmax(test_model_output)
# 创建精度函数
def get_accuracy(logits,targets):
    batch_predictions = np.argmax(logits,axis=1)
    num_correct = np.sum(np.equal(batch_predictions,targets))
    return(100.*num_correct/batch_predictions.shape[0])
my_optimizer = tf.train.MomentumOptimizer(learning_rate,0.9)
train_step = my_optimizer.minimize(loss)
# Initialize Variable
init = tf.initialize_all_variables()
sess.run(init)
#------------------------------------------------------
# 开始训练
#------------------------------------------------------
train_loss = []
train_acc = []
test_acc= []
for i in range(generations):
    rand_index = np.random.choice(len(train_xdata),size=batch_size)
    rand_x = train_xdata[rand_index]
    rand_x = np.expand_dims(rand_x,3)
    rand_y = train_labels[rand_index]
    train_dict = {x_input:rand_x,y_target:rand_y}
    sess.run(train_step,feed_dict=train_dict)
    temp_train_loss,temp_train_preds = sess.run([loss,prediction],feed_dict=train_dict)
    temp_train_acc = get_accuracy(temp_train_preds,rand_y)
    if i %eval_every==0:
        eval_index = np.random.choice(len(test_xdata),size=evalution_size)
        eval_x = test_xdata[eval_index]
        eval_x = np.expand_dims(eval_x,3)
        eval_y = test_labels[eval_index]
        test_dict = {eval_input:eval_x,eval_target:eval_y}
        test_preds = sess.run(test_prediction,feed_dict=test_dict)
        temp_test_acc = get_accuracy(test_preds,eval_y)
        #记录打印
        train_loss.append(temp_train_loss)
        train_acc.append(temp_train_acc)
        test_acc.append(temp_test_acc)
        acc_and_loss = [(i+1),temp_train_loss,temp_train_acc,temp_test_acc]
        acc_and_loss = [np.round(x,2) for x in acc_and_loss]
        print('Generation#{}.Train Loss:{:.2f}.Train Acc(Test Acc):{:.2f}({:.2f})'.format(*acc_and_loss))




