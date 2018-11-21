import numpy as np
from random import shuffle
import math

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
  num_classes = W.shape[1]
  num_train = X.shape[0]
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #pass
  for i in range(num_train):
    scores = X[i].dot(W)

   
    scores -= np.max(scores)  #Avoid numerical instability

    exps = np.exp(scores)
    softmax = np.exp(scores[y[i]])/np.sum(exps)
    loss += -np.log(softmax)

    p = exps/np.sum(exps)
    for j in range(num_classes):
      if j == y[i]:
        dfk = p[j] - 1
      else:
        dfk = p[j]
      dW[:,j] += dfk * X[i]
    
      
  loss /= num_train
  dW /= num_train
  dW += 2 * reg * W
  loss += reg * np.sum(W * W)  
  
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################


  
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[1]
  num_train = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #pass

  scores = X.dot(W)
  
  scores -= np.max(scores, axis=1, keepdims=True)
  exp_scores = np.exp(scores)
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
  loss = np.sum(-np.log(probs[np.arange(num_train),y]))

  #d_pk = p_k - 1(y_i = k)
  dscores = probs
  dscores[range(num_train),y] -= 1

  dW = np.dot(X.T, dscores)
  
  loss /= num_train
  loss += reg * np.sum(W * W)

  dW /= num_train
  dW += 2 * reg * W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

