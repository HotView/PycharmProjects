from sklearn import svm
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np


def plot_hyperplane(clf,X,y,h = 0.02,draw_sv = True,title = 'hyperplane'):
   x_min,x_max = X[:,0].min()-1,X[:,0].max()+1
   y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
   xx,yy = np.meshgrid(np.arange(x_min,x_max,h),
                       np.arange(y_min,y_max,h))
   Z = clf.predict(np.c_[xx.ravel(),yy.ravel()])
   # Put the result into a color plot
   Z = Z.reshape(xx.shape)
   plt.contourf(xx,yy,Z,cmap = 'hot',alpha = 0.5)

   markers= ['o','s','^']
   colors = ['b','r','c']
   labels = np.unique(y)
   for label in labels:
       plt.scatter(X[y==label][:,0] , X[y==label][:, 1],c=colors[label],marker=markers[label] )
   if draw_sv:
       sv = clf.support_vectors_
       plt.scatter(sv[:,0],sv[:,1],c = 'y',marker='x')
X,y = make_blobs(n_samples=100,centers=2,random_state=0,cluster_std=0.3)

clf = svm.SVC(C= 1.0,kernel="linear")
clf.fit(X,y)

plt.figure(figsize=(12,4),dpi=144)
plot_hyperplane(clf,X,y,h = 0.01,title='Maximum Margin Hyperplan')
plt.show()