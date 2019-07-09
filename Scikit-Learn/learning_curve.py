import numpy as np
from sklearn.pipeline import Pipeline
#构造多项式
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
# 学习曲线的绘制和学习
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt


n_dots = 200
X = np.linspace(0,1,n_dots)
y = np.sqrt(X)+0.2*np.random.rand(n_dots)-0.1
X = X.reshape(-1,1)
y = y.reshape(-1,1)

def polynomial_model(degree = 1):
    polynomial_features = PolynomialFeatures(degree=degree,include_bias=False)
    linear_regression = LinearRegression()
    # 这是一个流水线，先增加多项式阶数，然后用线形回归算法拟合数据
    pipeline  = Pipeline([("polynomial_fearures",polynomial_features),("linear_regression",linear_regression)])
    return pipeline
def plot_learning_curve(estimator,title,X,y,ylim = None,cv= None,n_jobs = 1,
                        train_sizes = np.linspace(.1,1.0,5)):
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("training example")
    plt.ylabel("Score")
    train_sizes,train_scores,test_socres = learning_curve(estimator,X,y,cv=cv,n_jobs = n_jobs,train_sizes = train_sizes)
    train_scores_mean = np.mean(train_scores,axis = 1)
    train_scores_std = np.std(train_scores,axis=1)
    test_scores_mean = np.mean(test_socres,axis=1)
    test_scores_std = np.std(test_socres,axis=1)
    print(train_sizes,"train_size")
    plt.grid()
    plt.fill_between(train_sizes,train_scores_mean-train_scores_std,train_scores_mean+train_scores_std,
                     alpha = 0.1,color = 'r')
    plt.fill_between(train_sizes,test_scores_mean - test_scores_std, test_scores_mean+test_scores_std,
                     alpha = 0.1,color = 'g')
    plt.plot(train_sizes,train_scores_mean,'o--',color = 'r',label = "Train score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label="cross-validation score")
    plt.legend(loc= "best")
    return plt
cv = ShuffleSplit(n_splits=10,test_size=0.2,random_state=0)
print(cv,"cv")

title = ["Learning Curves(Under Fitting)","Learning curves","Learning Curves(Over Fitting)"]
degrees = [1,3,10]
plt.figure(figsize=(18,4),dpi=200)
for i in range(len(degrees)):
    plt.subplot(1,3,i+1)
    plot_learning_curve(polynomial_model(degrees[i]),title[i],X,y = y,cv= cv,ylim=(0.75,1.01))
plt.show()

