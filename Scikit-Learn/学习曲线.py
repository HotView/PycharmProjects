from sklearn.model_selection import learning_curve
from sklearn.datasets import load_digits
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
import sklearn
print(sorted(sklearn.metrics.SCORERS.keys()))
digits = load_digits()
X =digits.data
y = digits.target
train_sizes,train_loss,test_loss  = learning_curve(SVC(gamma=0.001),X,y,cv=10,train_sizes=[0.1,0.2,0.5,0.6,0.7,0.8,0.9,1])
print(train_loss.shape)
print(test_loss.shape)
train_loss_mean = -np.mean(train_loss,axis=1)
test_loss_mean = -np.mean(test_loss,axis=1)

plt.plot(train_sizes,train_loss_mean,'o-',color= 'r',label = "training")
plt.plot(train_sizes,test_loss_mean,'o-',color= 'g',label = "Cross-validation")
plt.show()
