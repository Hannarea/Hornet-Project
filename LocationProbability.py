# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 17:00:02 2021

@author: hreed
"""

import numpy as np

# This is intended to be used as a module from which location_probability can be called. 

def location_probability(L, l):
    """
    Calculates the probability that a hornet is at location l given
    the previous locations Li 
    
    Use the minimum distance from a previous siting as the 
    argument into the pdf 

    Parameters
    ----------
    L : an np array of locations that were previously verified, each row is a new location
    l : an np array the new location

    Returns
    -------
    a scalar between 0 and 1

    """
    # Calculates the minimum norm 
    d = norm(L[:,0], l)
    for i in len(L[:,0]):
        if (d > norm(L[:,i], l)):
            d = norm(L[:,i], l)
    return pdf(d)


def norm(x, y):
    """
    calculates the l2 norm 

    Parameters
    ----------
    x : an np array
    y : an np array

    Returns
    -------
    a scalar, the l2 norm

    """
    if len(x) != len(y):
        print('invalid arguments into norm. len(x) != len(y)')
        return 0
    
    return np.sum(np.square(x-y))
    
    
def pdf(x):
    """
    

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    a number between 0 and 1.

    """
    sig = 5
    return 1/sig/np.sqrt(2*np.pi)*np.exp(-1/2*(x/sig)**2)

    
    
    