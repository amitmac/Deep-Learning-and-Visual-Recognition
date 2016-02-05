import numpy as np
from LinearSVM import *
from SoftmaxLoss import *

class LinearClassifier(object):
    def __init__(self):
        self.W = None
        
    def train(self,X,Y,reg=1e2,learning_rate=1e-7,num_iters=100,batch_size=200):
        num_train,dim = X.shape
        num_classes = np.max(Y) + 1
        num_batches = X.shape[0]/batch_size
        if self.W is None:
            self.W = 0.001 * np.random.randn(dim,num_classes)
        loss_history = []
        
        for i in range(num_iters):
            for j in range(num_batches):
                X_batch = X[j*batch_size:(j+1)*batch_size]
                Y_batch = Y[j*batch_size:(j+1)*batch_size]
            
                loss, dW = self.loss(X_batch,Y_batch,reg)
                loss_history.append(loss)
            
                self.W += -learning_rate * dW
            
        return loss_history
        
    def loss(self,X,Y,reg):
        pass
    
    def predict(self,X):
        return np.argmax(X.dot(self.W),axis=1)
    
class LinearSVMClassifier(LinearClassifier):
    
    def loss(self,X,Y,reg):
        return svm_loss(self.W,X,Y,reg)
    
class SoftmaxClassifier(LinearClassifier):
    
    def loss(self,X,Y,reg):
        return softmax_loss(self.W,X,Y,reg)
