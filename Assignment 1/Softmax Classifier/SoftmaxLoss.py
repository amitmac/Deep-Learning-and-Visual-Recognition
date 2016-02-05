import numpy as np

def softmax_loss_naive(W,X,Y,reg):
    """
    Naive Implementation (with loops)
    Inputs:
    - W: D x C array of weights (C is number of classes)
    - X: N x D array of input data (D is dimension of data)
    - y: N x 1 array of labels y[i] = c means that X[i] has label c, 
         where 0 <= c < C
    - reg: regularization strength
    
    Returns:
    - loss as single float
    - gradient with respect to weights W
    """
    dW = np.zeros(W.shape)
    (num_train,num_dim) = X.shape
    num_classes = W.shape[1]
    loss = 0.
    
    for i in xrange(num_train):
        scores = X[i].dot(W)
        scores -= np.max(scores)
        for j in xrange(num_classes):
            num = np.exp(scores[j])
            den = np.sum(np.exp(scores))
            loss += -np.log(num/den)
            if j == Y[i]:
                dW[:,j] += X[i] * (num/den - 1.)
            else:
                dW[:,j] += X[i] * (num/den)
        
    loss /= num_train
    loss += reg * (W ** 2).sum()
    
    dW /= num_train
    dW += 0.5 * reg * W
    
    return loss, dW

def softmax_loss(W,X,Y,reg):
    """
    Vectorized Implementation
    """
    dW = np.zeros(W.shape)
    (num_train,num_dim) = X.shape
    num_classes = W.shape[1]
    loss = 0.
    
    scores = X.dot(W)
    scores -= np.max(scores,axis=1)[:,None]
    num = np.exp(scores)
    den = np.sum(np.exp(scores),axis=1)
    
    margin = num/den[:,None]
    margin[range(num_train),Y] -= 1
    
    loss = np.sum(-np.log(num/den[:,None]))/num_train
    loss += reg * (W**2).sum()
    
    #Also multiply by (P(y|x) - 1)
    C = X.T.dot(margin)
    C /= num_train
    dW = C + reg * W
    
    return loss, dW