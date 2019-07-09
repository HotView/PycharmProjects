import numpy as np
import matplotlib.pyplot as plt
class Neural_Network():
    # J = 1/2*(y-yHat)^2
    def __init__(self):
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3

        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)
    def forward(self,X):
        #z2 = X*W1
        #a2 = f(z2)
        #z3 = a2*W2
        self.z2 = np.dox(X,self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2,self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat
    def costFunction(self,X,y):
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)
        return J
    def costFunctionPrime(self,X,y):
        # dJdW2 = a2.T*delta3
        # delta3 = -(y-yHat)f'(z3)
        self.yHat = self.forward(X)

        delta3 = np.multiply(-(y-self.yHat),self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T,delta3)

        delta2 = np.dot(delta3,self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T,delta2)

        return dJdW1,dJdW2
    def sigmoid(self,z):
        # a2 = f(Z2)
        #y = f(Z3)
        return 1/(1+np.exp(-z))
    def sigmoidPrime(self,z):
        # Derivative of Sigmoid Function
        return np.exp(-z)/((1+np.exp(-z))**2)

NN = Neural_Network()
testValues = np.arange(-5,5,0.01)
plt.plot(testValues,NN.sigmoid(testValues),linewidth = 2)
plt.plot(testValues,NN.sigmoidPrime(testValues),linewidth =2)
plt.grid(1)
plt.legend(['sigmod','sigmoidPrime'])
plt.show()
X = None
y = None
cost1 = NN.costFunctionPrime(X,y)
dJdW1,dJdW2 = NN.costFunctionPrime(X,y)
scaler = 3
NN.W1 = NN.W1-scaler*dJdW1
NN.W2 = NN.W2-scaler*dJdW2
cost2 = NN.costFunctionPrime()
