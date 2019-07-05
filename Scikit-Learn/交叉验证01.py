# 选定参数的程序测试
from sklearn.model_selection import validation_curve
from sklearn.datasets import load_digits
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
import sklearn
print(sorted(sklearn.metrics.SCORERS.keys()))
digits = load_digits()
X =digits.data
y = digits.target
param_range = np.logspace(-6,-2.3,20)
train_loss,test_loss  = validation_curve(SVC( ),X,y,param_name='gamma',param_range=param_range,cv=10)

print(train_loss.shape)
print(test_loss.shape)
train_loss_mean = -np.mean(train_loss,axis=1)
test_loss_mean = -np.mean(test_loss,axis=1)

plt.plot(param_range,train_loss_mean,'o-',color= 'r',label = "training")
plt.plot(param_range,test_loss_mean,'o-',color= 'g',label = "Cross-validation")
plt.xlabel("gamma")
plt.ylabel("loss")
plt.legend(loc= "best")
plt.show()
