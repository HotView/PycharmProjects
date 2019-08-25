import d2lzh as d2l
from d2lzh import resnet18
from mxnet import autograd,gluon,init
from mxnet.gluon import data as gdata,loss as gloss ,nn
import os
import shutil
import time
demo = False

def read_label_file(data_dir,label_file,train_dir,valid_ratio):
    with open(os.path.join(data_dir,label_file),'r') as f:
        lines = f.readlines()[1:]
        print(lines)
        tokens= [l.rstrip().split(',') for l in lines]
        idx_label = dict(((int(idx),label) for idx,label in tokens))
    labels = set(idx_label.values())
    n_train_valid = len(os.listdir(os.path.join(data_dir,train_dir)))
    n_train = int(n_train_valid*(1-valid_ratio))
    assert  0<n_train<n_train_valid
    return n_train//len(labels),idx_label
def mkdir_if_not_exist(path): # 本函数已保存在d2lzh包中⽅便以后使⽤
    if not os.path.exists(os.path.join(*path)):
        os.makedirs(os.path.join(*path))
def reorg_train_valid(data_dir,train_dir,input_dir,n_train_per_label,idx_label):
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir,train_dir)):
        idx = int(train_file.split('.')[0])
        print(idx)
        label = idx_label[idx]
        mkdir_if_not_exist([data_dir,input_dir,'train_valid',label])
        shutil.copy(os.path.join(data_dir, train_dir, train_file),
                    os.path.join(data_dir, input_dir, 'train_valid', label))
        if label not in label_count or label_count[label]<n_train_per_label:
            mkdir_if_not_exist([data_dir, input_dir, 'train', label])
            shutil.copy(os.path.join(data_dir, train_dir, train_file),
                        os.path.join(data_dir, input_dir, 'train', label))
            label_count[label] = label_count.get(label,0)+1
        else:
            mkdir_if_not_exist([data_dir,input_dir,'valid',label])
            shutil.copy(os.path.join(data_dir,train_dir,train_file),
                        os.path.join(data_dir,input_dir,'valid',label))
def reorg_test(data_dir,test_dir,input_dir):
    mkdir_if_not_exist([data_dir,input_dir,"test","unknown"])
    for test_file in os.listdir(os.path.join(data_dir,test_dir)):
        shutil.copy(os.path.join(data_dir,test_dir,test_file),
                    os.path.join(data_dir,input_dir,'test','unknown'))
def reorg_cifar10_data(data_dir,label_file,train_dir,test_dir,input_dir,valid_ratio):
    n_train_per_label,idx_label = read_label_file(data_dir,label_file,train_dir,valid_ratio)
    reorg_train_valid(data_dir,train_dir,input_dir,n_train_per_label,idx_label)
    reorg_test(data_dir,test_dir,input_dir)

data_dir = "E:/Download/kaggle_cifar10"
train_dir = "train"
label_file = "trainLabels.csv"
input_dir = "input"
valid_ratio = 0.1
batch_size = 256
#n_train_per_label,idx_label = read_label_file(data_dir,label_file,train_dir,valid_ratio)
#print(n_train_per_label)
#print(len(idx_label),"idx_label")
#reorg_train_valid(data_dir,train_dir,input_dir,n_train_per_label,idx_label)
transform_train = gdata.vision.transforms.Compose([gdata.vision.transforms.Resize(40),
                                                   gdata.vision.transforms.RandomResizedCrop(32,scale=(0.64, 1.0),
                                                    ratio=(1.0, 1.0)),
                                                    gdata.vision.transforms.RandomFlipLeftRight(),
                                                    gdata.vision.transforms.ToTensor(),
                                                    # 对图像的每个通道做标准化
                                                    gdata.vision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                                                                      [0.2023, 0.1994, 0.2010])])
transform_test = gdata.vision.transforms.Compose([gdata.vision.transforms.ToTensor(),
                                                  gdata.vision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                                                                    [0.2023, 0.1994, 0.2010])])
train_ds = gdata.vision.ImageFolderDataset(os.path.join(data_dir, input_dir, 'train'), flag=1)
valid_ds = gdata.vision.ImageFolderDataset(os.path.join(data_dir, input_dir, 'valid'), flag=1)
train_valid_ds = gdata.vision.ImageFolderDataset(os.path.join(data_dir, input_dir, 'train_valid'), flag=1)
test_ds = gdata.vision.ImageFolderDataset(os.path.join(data_dir, input_dir, 'test'), flag=1)

train_iter = gdata.DataLoader(train_ds.transform_first(transform_train),batch_size, shuffle=True, last_batch='keep')
valid_iter = gdata.DataLoader(valid_ds.transform_first(transform_test),batch_size, shuffle=True, last_batch='keep')
train_valid_iter = gdata.DataLoader(train_valid_ds.transform_first(transform_train), batch_size, shuffle=True, last_batch='keep')
test_iter = gdata.DataLoader(test_ds.transform_first(transform_test),batch_size, shuffle=False, last_batch='keep')
## 定义模型 ResNet 模型
def get_net(ctx):
    num_classes = 10
    net = resnet18(num_classes)
    net.initialize(ctx=ctx, init=init.Xavier())
    return net
loss = gloss.SoftmaxCrossEntropyLoss()
## 定义训练函数
def train(net,train_iter,valid_iter,num_epochs,lr,wd,ctx,lr_period,lr_decay):
    trainer = gluon.Trainer(net.collect_params(),'sgd',{"learning_rate":lr,'momentum': 0.9, 'wd': wd})
    for epoch in range(num_epochs):
        train_l_sum,train_acc_sum,n,start = 0.0,0.0,0,time.time()
        if epoch>0 and epoch%lr_period==0:
            trainer.set_learning_rate(trainer.learning_rate*lr_decay)
        for X,y in train_iter:
            print(X,"---------------")
            print(len(X),"lesx")
            print(y,"#############")
            print(len(y))
            y = y.astype("float32").as_in_context(ctx)
            with autograd.record():
                y_hat = net(X.as_in_context(ctx))
                l = loss(y_hat,y).sum()
            l.backward()
            trainer.step(batch_size)
            train_l_sum+=l.asscalar()
            train_acc_sum+=(y_hat.argmax(axis=1)==y).sum().asscalar()
            n+=y.size
        time_s = "time %.2f sec"%(time.time()-start)
        if valid_iter is not None:
            valid_acc = d2l.evaluate_accuracy(valid_iter,net,ctx)
            epoch_s = ("epoch %d,loss %f,train_acc %f,valid_acc %f,"
                       %(epoch+1,train_l_sum/n,train_acc_sum/n,valid_acc))
        else:
            epoch_s = ("epoch %d,loss %f,train_acc %f,"
                       %(epoch+1,train_l_sum/n,train_acc_sum/n))
        print(epoch_s+time_s+',lr'+str(trainer.learning_rate))
print("dshjkdf")
ctx,num_epochs,lr,wd =d2l.try_gpu(),1,0.1,5e-4
lr_period,lr_decay,net = 80,0.1,get_net(ctx)
net.hybridize()
train(net,train_iter,valid_iter,num_epochs,lr,wd,ctx,lr_period,lr_decay)
