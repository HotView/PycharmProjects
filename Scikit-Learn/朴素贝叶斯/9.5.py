from time import time
from sklearn.datasets import load_files

print("loading train datasets")
t = time()
news_train = load_files('datasets/379/train')
print("summary ;{0} document in {1} categories.".format(len(news_train.data),len(news_train.target_names)))
print("done in {0} seconds".format(time()-t))

from sklearn.feature_extraction.text import TfidfVectorizer
print("vectorizing train datasets...")
t = time()
vectorizer = TfidfVectorizer(encoding='latin-1')
X_train = vectorizer.fit_transform(d for d in news_train.data)
print("n_sample:%d,n_features:%d"%X_train.shape)
print("number of non-zeor features in sample[{0}]:{1}".format(news_train.filenames[0],X_train[0].getnnz()))
print("done in {0}seconds".format(time()-t))
from sklearn.naive_bayes import MultinomialNB
print("training models ...".format(time()-t))
t =time()
y_train = news_train.target
clf = MultinomialNB(alpha=0.0001)
clf.fit(X_train,y_train)
train_score = clf.score(X_train,y_train)
print("train score:{0}".format(train_score))
print("done in {0} seconds".format(time()-t))
print("loading test datasets...")
t = time()
news_test = load_files('datasets/379/test')
print("summary:{0}document in {1} categories".format(len(news_test.data),len(news_test.target_names)))

print("vectorizing test dataset...")
t = time()
X_test = vectorizer.transform((d for d in news_test.data))
y_test = news_test.target
print("n_sample:%d,n_features:%d"%X_test.shape)
print("number of non-zeor features in sample[{0}]:{1}".format(news_test.filenames[0],X_test[0].getnnz()))
pred = clf.predict(X_test[0])
print("predit:{0} is in category {1}".format(news_test.filenames[0],news_test.target_names[pred[0]]))
print("predit:{0} is in category {1}".format(news_test.filenames[0],news_test.target_names[news_test.target[0]]))
from sklearn.metrics import classification_report
print("classfication report on test set for classifier")
print(clf)
print(classification_report(y_test,pred,target_names=news_test.target_names))