from mxnet import ndarray as nd
from mxnet import autograd
num_inputs = 2
num_examples = 1000

true_w = [2,-3.4]
true_b = 4.2

X = nd.random_normal(shape=(num_examples,num_inputs))
y = true_w[0]*X[:,0]+true_w[1]*X[:,1]+true_b
y+=  0.01*nd.random_normal(shape=y.shape)
print(X[0:10],y[0:10])
import random
batch_size = 10
def data_iter():
    idx = list(range(num_examples))
    random.shuffle(idx)
    for i in range(0,num_examples,batch_size):
        j = nd.array(idx[i:min(i+batch_size,num_examples)])
        yield nd.take(X,j),nd.take(y,j)
for data,label in data_iter():
    print(data,label)
## 初始化模型
w = nd.random_normal(shape=(num_inputs,1))
b = nd.zeros((1,))
params = [w,b]
for param in params:
    param.attach_grad()
def net(X):
    return nd.dot(X,w)+b
def square_loss(y_hat,y):
    return (y_hat-y.reshape(y_hat.shape))**2
def SGD(params,learn_rate):
    for param in params:
        param[:] = param-learn_rate*param.grad
## 训练模型
epochs = 5
learning_rate = 0.001
for e in range(epochs):
    total_loss = 0
    i = 0
    for data,label in data_iter():
        print(i)
        i = i+1
        with autograd.record():
            output = net(data)
            loss = square_loss(output,label)
        loss.backward()
        SGD(params,learning_rate)
        total_loss+=nd.sum(loss).asscalar()
    print(e,total_loss/num_examples)

