from sklearn import svm
from sklearn import datasets

clf = svm.SVC()
iris = datasets.load_iris()
X,y = iris.data,iris.target
# clf.fit(X,y)
import pickle
# with open("clf.pickle","wb") as f:
#    pickle.dump(clf,f)
with open("clf.pickle","rb") as f:
    clf2 = pickle.load(f)
    print(clf2.predict(X[0:1]))
    print(y[0:1])

