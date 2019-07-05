from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
import numpy as np

centers = [[-2,2],[2,2],[0,4]]
X,y = make_blobs(n_samples=60,centers = centers,random_state=0,cluster_std=0.60)
print(y)
plt.figure(figsize=(16,10),dpi=50)
c = np.array(centers)
plt.scatter(X[:,0],X[:,1],c = y,s = 100,cmap='cool')
plt.scatter(c[:,0],c[:,1],s = 100,marker='^',c = 'orange')
#plt.show()
from sklearn.neighbors import KNeighborsClassifier
k = 5
clf = KNeighborsClassifier(n_neighbors=k)
clf.fit(X,y)
X_sample =np.array([0,2]).reshape(-1,2)
y_sample = clf.predict(X_sample)
neighours = clf.kneighbors(X_sample,return_distance=False)
print(neighours)
print(X_sample)

plt.scatter(X_sample[0,0],X_sample[0,1],marker='x',c = y_sample[0],s =100,cmap="cool")
for i in neighours[0]:
    plt.plot([X[i][0],X_sample[0,0]],[X[i][1],X_sample[0,1]],'k--',linewidth = 0.6)
plt.show()
