import numpy as np

def svm_loss_naive(W,X,y,reg):
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
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.
    
    for i in xrange(num_train):
        scores = X[i].dot(W)
        for j in xrange(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - scores[y[i]] + 1.
            if margin > 0:
                loss += margin
                dW[:,j] += X[i,:]
                dW[:,y[i]] -= X[i,:]
    
    loss /= num_train
    dW /= num_train
    
    loss += 0.5 * reg * (W**2).sum()
    dW += reg * W
    
    return loss,dW

def svm_loss(W,X,y,reg):
    """
    Vectorized Implementation
    """
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.
    dW = np.zeros(W.shape)
    
    score = X.dot(W)
    
    y_pred = score[range(score.shape[0]),y]
    
    margins = score - y_pred[:,None] + 1.
    margins[range(score.shape[0]),y] = 0
    margins = np.maximum(np.zeros(margins.shape),margins)
    
    loss = np.sum(margins)
    loss /= num_train
    loss += 0.5 * reg * (W**2).sum()
    
    non_zeros_count = (margins > 0).sum(axis=1)
    
    multiplier = margins
    multiplier[margins > 0] = 1
    multiplier[range(num_train),y] = -1. * non_zeros_count 
    
    dW = X.T.dot(multiplier)
    dW /= num_train
    dW += reg * W
    
    return loss,dW
