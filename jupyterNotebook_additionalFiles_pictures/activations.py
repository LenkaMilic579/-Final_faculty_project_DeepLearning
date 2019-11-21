import numpy as np

def stable_softmax(Z):
    """
    Implement the SOFTMAX function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; 
    stored for computing the backward pass efficiently
    """  
    stabilize = Z - np.max(Z)  # shifts all of elements in the vector to negative to zero
    exps = np.exp(stabilize) # negatives with large exponents saturate to zero rather than the infinity
    suma = np.sum(exps)
    result = np.divide(exps, suma) # avoiding overflowing and resulting in nan
    catch = Z       
    
    assert(result.shape == Z.shape)
    return result, catch

def softmax_backward(dA, AL, cache):
    Z = cache
    m, n = AL.shape
    ones =  np.ones((n, m))
    matrix = np.matmul(AL, ones) * (np.identity(m) - np.matmul(np.ones((m, n)), AL.T))
    dZ = np.matmul(matrix, dA)
    assert (dZ.shape == Z.shape)
    return dZ

def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; 
    stored for computing the backward pass efficiently
    """                
    A = np.maximum(0.0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache

def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)    
    return dZ