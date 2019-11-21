import numpy as np
import math
from helpers_conv import *
from helpers_fc import *
from helpers1 import solve_fc, solve_conv

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def cross_entropy_loss(AL, Y):
    
    """
    Implement the cost function
    
    Arguments:
    a3 -- post-activation, output of forward propagation size of [num_of_atributes x num_of_examples ]
    Y -- "true" labels vector, same shape as a3
    
    Returns:
    cost - value of the cost function
    """
    m = Y.shape[1]
    #mnozenje = np.multiply(Y, np.log(AL))
    #negsum = -np.sum(mnozenje)
    #final = negsum / m
    return 1./m * -1 * np.sum(Y * (AL + (-np.max(AL) - np.log(np.sum(np.exp(AL-np.max(AL)))))))

def predict(X, y, parameters, layers_dims_conv, channels, nc):
    """   
    X -- train/test data of shape (#examples x dim(image))
    y -- train/test data of shape (#examples x 1)
    parameters -- parameters of the trained model    
    Returns:
    p -- predictions for the given dataset X of shape (#examples x 1)    
    """    
    m = X.shape[0]
    hparameters1 = {"pad" : 2, "stride": 3}
    hparameters2 = {"stride" : 2, "f": 2}
    hparameters3 = {"pad" : 1, "stride": 3}
    hparameters4 = {"stride" : 2, "f": 2}    
    
    parameters_conv = solve_conv(parameters, layers_dims_conv, channels, nc)    
    # Forward propagation conv 
    P2, cache_pool2, cache_pool1, cache_conv1, cache_conv2, c1, c2 = Convolutional_Forward(X, parameters_conv, hparameters1, hparameters2, hparameters3, hparameters4)
    
    # Forward propagation FC
    FC1 = P2.reshape(P2.shape[0], -1).T
    
    # Get parameters for FC network, parameters_fc
    parameters_fc = solve_fc(parameters, layers_dims_conv)    
    # Forward propagation FC
    A3_fc, cache_fc = L_model_forward(FC1, parameters_fc)
    
    a, b = A3_fc.shape
    p = np.zeros((a, b))    
    # convert probas to 0/1 predictions
    tmp = np.argmax(A3_fc, axis=0)
    for i in range(0, b):
        p[tmp[i],i] = 1
    p = p.T
    print("Accuracy: "  + str(np.mean((p[:,:] == y[:,:]))))    
    return p