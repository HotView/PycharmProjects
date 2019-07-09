# 手写板的数据
from sklearn import datasets
from matplotlib import pyplot as plt
from sklearn.model_selection import  train_test_split
from sklearn import svm
from sklearn.externals import joblib
digits = datasets.load_digits()
images_and_labels = list(zip(digits.images,digits.target))
plt.figure(figsize=(8,6),dpi=200)
for index,(image,label) in enumerate(images_and_labels[:8]):
    plt.subplot(2,4,index+1)
    plt.axis("off")
    plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
    plt.title('Digit:%i'%label,fontsize = 20)
print("shape of raw image data:{0}".format(digits.images.shape))
print("shape of data:{0}".format(digits.data.shape))
print("shape of target:{0}".format(digits.target.shape))
#plt.show()
Xtrain,Xtest,Ytrain,Ytest = train_test_split(digits.data,digits.target,test_size=0.2,random_state=2)
clf = svm.SVC(gamma=0.001,C =100.)
clf.fit(Xtrain,Ytrain)
score = clf.score(Xtest,Ytest)
p = clf.get_params()
print(score)
#print(clf.coef_,"fdlksf")
fig,axes =plt.subplots(4,4,figsize=(8,8))
fig.subplots_adjust(hspace =0.1,wspace = 0.1)
Ypred = clf.predict(Xtest)
for i,ax in enumerate(axes.flat):
    ax.imshow(Xtest[i].reshape(8,8),cmap=plt.cm.gray_r,interpolation='nearest')
    ax.text(0.05,0.05,str(Ypred[i]),fontsize = 32,transform = ax.transAxes,color = 'green' if Ypred[i]==Ytest[i] else 'red')
    ax.text(0.8, 0.05, str(Ytest[i]), fontsize=32, transform=ax.transAxes, color='black')
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()
print(clf)
joblib.dump(clf,'digits_svm.pkl')