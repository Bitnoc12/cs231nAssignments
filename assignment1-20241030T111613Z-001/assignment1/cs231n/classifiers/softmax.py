from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0] # 49000
    num_classes = W.shape[1] # 10

    for i in range(num_train):
      tmp = np.dot(X[i], W) # 1x3073 x 3073x10
      exp_tmp = np.exp(tmp) # 1x10
      # 对每个分类求scores概率
      scores = exp_tmp / np.sum(exp_tmp) # get the scores in all classification
      # 对每个x进行loss计算
      loss += -np.log(scores[y[i]]) 

      # 计算梯度
      for k in range(num_classes):
        # 获取当前分类的概率
        p_k = scores[k]
        if k == y[i]:
          dW[:,k] += (p_k - 1) * X[i] # 正确分类的梯度
        else:
          dW[:,k] += p_k * X[i] 

    # divide N
    loss /= num_train
    dW /= num_train
    # add regularization terms
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]

    tmp = np.dot(X, W)
    exp = np.exp(tmp)
    prob = exp / np.sum(exp, axis=1, keepdims=True)
    
    # cal loss
    loss += np.sum(-np.log(prob[range(num_train), y]))

    # cal grad : P-1 就可以得到梯度
    prob[range(num_train), y] -= 1 
    dW += np.dot(X.T, prob)
    # print(X.shape)
    # print(prob.shape)

    loss /= num_train
    dW /= num_train
    
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
