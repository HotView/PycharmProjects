# 单层神经网络
import matplotlib.pyplot as plt
from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd

def transform(data,label):
    return data.astype('float32')/255,label.astype('float32')
#hahahtest = gluon.data.vision.transforms.Resize(96)
mnist_train  = gluon.data.vision.FashionMNIST(train=True,transform=transform)
mnist_test = gluon.data.vision.FashionMNIST(train=False,transform=transform)
data,label = mnist_train[0]

def show_image(images):
    n = images.shape[0]
    _,figs = plt.subplots(1,n,figsize = (15,15))
    for i in range(n):
        figs[i].imshow(images[i].reshape((28,28)).asnumpy(),'gray')
        figs[i].axes.get_xaxis().set_visible(False)
        figs[i].axes.get_yaxis().set_visible(False)
    plt.show()
def get_text_labels(label):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in label]
data,label = mnist_train[0:9]
#show_image(data)
#print(get_text_labels(label))
batch_size = 256
train_data = gluon.data.DataLoader(mnist_train,batch_size,shuffle=True)
test_data = gluon.data.DataLoader(mnist_test,batch_size,shuffle=False)
num_inputs = 784
num_outputs = 10

W = nd.random_normal(shape=(num_inputs,num_outputs))
b = nd.random_normal(shape=num_outputs)
params = [W,b]
for param in params:
    param.attach_grad()
from mxnet import nd
def softmax(X):
    exp = nd.exp(X)
    partition = exp.sum(axis=1,keepdims=True)
    return exp/partition
def net(X):
    return softmax(nd.dot(X.reshape((-1,num_inputs)),W)+b)
## 交叉损失熵
def cross_entroy(y_hat,y):
    return -nd.pick(nd.log(y_hat),y)
def accuracy(output,label):
    return nd.mean(output.argmax(axis =1)==label).asscalar()
def SGD(params,learn_rate):
    for param in params:
        param[:] = param-learn_rate*param.grad
def evaluate_accuracy(data_iter,net):
    acc = 0.0
    for data,label in data_iter:
        output = net(data)
        acc+=accuracy(output,label)
    return acc/len(data_iter)
print(evaluate_accuracy(test_data,net))
learining_rate = 0.1
for epoch in range(5):
    train_loss = 0.
    train_acc = 0.
    for data,label in train_data:
        print(label)
        with autograd.record():
            output = net(data)
            loss = cross_entroy(output,label)
        loss.backward()
        SGD(params,learining_rate/batch_size)
        train_loss+=nd.mean(loss).asscalar()
        train_acc+=accuracy(output,label)
    test_acc = evaluate_accuracy(test_data,net)
    print(epoch,train_loss/len(train_data),train_acc/len(train_data),test_acc)
data,label = mnist_test[0:9]
show_image(data)
print(get_text_labels(label))
predict_labels = net(data).argmax(axis=1)
print(get_text_labels(predict_labels.asnumpy()))
