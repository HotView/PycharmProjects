from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


iris = load_iris()
X = iris.data
y = iris.target
k_range = range(1,31)
k_sore = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn,X,y,cv=10,scoring='accuracy' )
    k_sore.append(scores.mean())
plt.plot(k_range,k_sore)
plt.xlabel("value of K for KNN")
plt.ylabel("Cross-Validated Accuracy")
plt.show()



