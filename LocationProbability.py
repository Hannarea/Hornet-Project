# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 17:00:02 2021

@author: hreed
"""

import numpy as np

# This is intended to be used as a module from which location_probability can be called. 


def build_location_probabilities_column(data):
    '''
    build a column of location probabilities for the data used to train the model
    assumed that the data is in chronological order with row 0 being oldest case on 9/19/2019

    Parameters
    ----------
    data : an matrix of previous sightings location infomation- given in format
            [lab status, latitude, longitude]

    Returns
    -------
    a vector of the location probabilities for the data to train the model 

    '''
    # This is the first positive case in the data! date 9/19/2019
    prior = np.array([49.1494, -123.943])
    location_probabilities = np.array([1]) # We know that the first positive case is positive
    
    for i in range(len(data)): # for each row in the data
        # Calculate the location_probability and add to our vector
        p = location_probability(prior, convert_coordinates(data[i, 1:]))
        location_probabilities = np.append(location_probabilities, p)
        # add the location to priors if it was actually positive
        if data[i, 0] == 'Positive ID':
            prior = np.vstack([prior, data[i, 1:]])
        # repeat!
    return location_probabilities
    
    

def convert_coordinates(location):
    '''
    converts location

    Parameters
    ----------
    location : an np.array with the latitude and longitude

    Returns
    -------
    an np.array with the new location values (in what coordiate system?)

    '''
    return location # Replace this

    
    

def location_probability(prior, new):
    """
    Calculates the probability that a hornet is at location l given
    the previous locations Li 
    
    Use the minimum distance from a previous siting as the 
    argument into the pdf 

    Parameters
    ----------
    prior : an np array of locations that were previously verified, each row is a new location
    new : an np array the new location

    Returns
    -------
    a scalar between 0 and 1

    """
    # Calculates the minimum norm 
    d = norm(prior[:,0], new)
    for i in len(prior[:,0]):
        if (d > norm(prior[:,i], new)):
            d = norm(prior[:,i], new)
    return pdf(d)


def add_new_location(prior, new, ID, keep = 'Positive ID'):
    '''
    
    add the new location to the prior locations IF it is a positive ID 

    Parameters
    ----------
    prior : array of prior locations that are already known
    new : location under consideration
    ID : the value in the column 'Lab Status'

    Returns
    -------
    Updated prior array

    '''
    
    if ID == keep:
        new_row = np.array([new[0], new[1]])
        return np.vstack([prior, new_row])
    return prior


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

    
    
    