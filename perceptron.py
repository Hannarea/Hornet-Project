# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 15:40:37 2021

@author: hreed
"""

import numpy as np

# This code is for one perceptron. 

def activation_function(y):
    """
    Activation function using the sigmoid function.
    
    Parameters
    ----------
    y : a scalar, the dot product of the vector of inputs 
        and weights for this perceptron.

    Returns
    -------
    A value between 0 and 1 - the predicted value y_hat.

    """
    return 1/(1+np.exp(-y))


def cost_function(actual, predicted):
    """
    calculates the sum of the loss (i=0,...,n)
    Loss function uses the mean squared error (for each i)
    
    Parameters
    ----------
    actual : np.array of true values .
    predicted : np.array of predicted values from the activation function.

    Returns
    -------
    a scalar, measurment of loss.

    """
    return 1/len(actual)*np.sum(np.square(actual - predicted))

def cost_partial_w(actual, predicted, b, x, w):
    """
    computes the partial derivative of the cost function with respect to 
    the weight vector w
    
    Parameters
    ----------
    actual : np.array of true values .
    predicted : np.array of predicted values from the activation function.
    b : current bias, b in wx+b
    x : np.array of the inputs
    w : the vector of the weights

    Returns
    -------
    np.array of the partial derivative of the cost function with respect 
    to the weights wi

    """
    
    # The length of the vector of partials will be the same as the number of inputs.
    partials = np.zeros(len(x)) 
    for i in range(len(x)):
        partials[i] = 2/len(x) * cost_function(actual, predicted) * activation_function(w.T@x+b)*(1-activation_function(w.T@x+b))*x[i]
    return partials


def cost_partial_b(actual, predicted, b, x, w):
    """
    computes the partial derivative of the cost function with 
    respect to the bias b
    
    Parameters
    ----------
    actual : np.array of true values .
    predicted : np.array of predicted values from the activation function.
    b : current bias, b in wx+b
    x : np.array of the inputs
    w : the vector of the weights

    Returns
    -------
    scalar, the partial derivative of the cost function with respect
    to the bias b

    """
    return 2/len(x) * cost_function(actual, predicted) * activation_function(w.T@x+b)*(1-activation_function(w.T@x+b))
    
    
# We first need to determing the weights and the bias using the gradient descent method

# we start with random values for the weights: 
    


# Now, we can use these weights and bias to give a prediction on some data: 
    
    
    
    
    
    
    
    
    