# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 17:00:02 2021

@author: hreed
"""

import numpy as np
import matplotlib.pyplot as plt

# This is intended to be used as a module from which location_probability can be called. 





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
    #[lat,long]
    #P is a coordinate for a reported sighting
    #d(P,Q)
    #minimize d(P,Q) where Q is confirmed sighting locations

    
    return location # Replace this




    
    
def pdf(x):
    """
    

    Parameters
    ----------
    x : scalar

    Returns
    -------
    a number between 0 and 1.

    """
    D = 1
    return 1- x/D

x = np.linspace(0,1,20)
plt.plot(x, pdf(x), 'ro')
plt.plot(.5, .5, 'bo')
plt.show()
    

def norm (x, y):
    return np.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2)

def location_probability(prior, new):
    """
    Calculates the probability that a hornet is at location new given
    the previous locations in prior
    
    Use the minimum distance from a previous siting as the 
    argument into the pdf 

    Parameters
    ----------
    prior : an nx2 np array of locations that were previously verified, each row is a new location
    new : an 1x2 np array the new location

    Returns
    -------
    a scalar between 0 and 1

    """
    # Calculates the minimum norm
    
    d = norm(prior[0,:], new)
    for i in range(len(prior)):
        if (d > norm(prior[i, :], new)):
            d = norm(prior[i, :], new)
    return pdf(d)



def build_location_probabilities_vector(data, prior = np.array([[49.1494, -123.943]])):
    '''
    build a column of location probabilities for the data used to train the model
    assumed that the data is in chronological order with row 0 being oldest case

    Parameters
    ----------
    data : an matrix of previous sighting location infomation- given in format
            [lab status, latitude, longitude] including the sightings in prior
    prior : an nx2 matrix of n previous locations, default is location of the first positive 
            case in the data on 9/19/2019

    Returns
    -------
    a vector of the location probabilities for the data 

    '''
    
    #Hello Hanna
    
    location_probabilities = np.ones(len(prior)) # We know that the prior known cases were true
    k = len(prior)
    for i in range(k, len(data)): # for each row in the data
        # Calculate the location_probability and add to our vector
        p = location_probability(prior, convert_coordinates(data[i, 1:]))
        print(p)
        location_probabilities = np.append(location_probabilities, p)
        # add the location to priors if it was actually positive
        if data[i, 0] == 1: # 1 means 'Positive ID', we converted these already
            prior = np.vstack([prior, data[i, 1:]])
        # repeat!
    return location_probabilities

# Example - the red dots have high probability and the blue have low probability        

prior = np.array([[0,0]])
data = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, -.5, .5],
    [0, .5, -.5],
    [0, -1, 0],
    [0, -.2, 0],
    [0, .2, 0],
    [0, 1, 0],
    [0, -.5, -.5],
    [0, .5, -.5],
    [0, 0, -1],
    [0, 0, .2],
    [0, 0, -.2],
    [1, .5, .5],
    [0, .7, .6], # Here, we have reds because of the true case right before
    [0, .6, .4],
    [0, .4, .55]
    ])
v = build_location_probabilities_vector(data, prior = prior)
print(v)

# the small dots show where the points like on the pdf. they should decrese as the points get farther from the origin in this example
for i in range(len(data)):
    if v[i] > 0.5:
        plt.plot(data[i,1], data[i,2], 'ro')
    else:
        plt.plot(data[i,1], data[i,2], 'bo')

    

    