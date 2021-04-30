# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 17:00:02 2021

@author: hreed
"""

import numpy as np

# This is intended to be used as a module from which location_probability can be called. 

def arc_length(location1, location2):
    '''
    returns the distance between location 1 and location 2 given in latitude, longitude

    Parameters
    ----------
    location1 : 1x2 array of latitude, longitude
    location1 : 1x2 array of latitude, longitude

    Returns
    -------
    scarlar, the distance between the points in kilometers

    '''
    R = 6367.4445 # assumed radius of the earth, in km
    p1 = location1[0] * np.pi / 180
    p2 = location2[0] * np.pi / 180
    l1 = location1[1] * np.pi / 180
    l2 = location2[1] * np.pi / 180
    
    d1 = (p2-p1)/2
    d2 = (l2-l1)/2
    
    guts = np.sin(d1)**2 + np.cos(p1)* np.cos(p2) * np.sin(d2)**2
    return 2*R*np.arcsin(np.sqrt(guts))

    
def pdf(x):
    return 1 - x/600


def norm (x, y):
    return np.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2)


def location_probability(prior, new, distance):
    """
    Calculates the probability that a hornet is at location new given
    the previous locations in prior
    
    Use the minimum distance from a previous siting as the 
    argument into the pdf 

    Parameters
    ----------
    prior : an nx2 np array of locations that were previously verified, each row is a new location
    new : an 1x2 np array the new location
    distance: the distance function to be used, arc_length, norm, etc. 

    Returns
    -------
    a scalar between 0 and 1

    """
    # Calculates the minimum norm
    d = distance(prior[0,:], new)
    for i in range(len(prior)):
        if (d > distance(prior[i, :], new)):
            d = distance(prior[i, :], new)
    
    return pdf(d)


def build_location_probabilities_vector(data, distance, prior = np.array([[49.1494, -123.943]])):
    '''
    build a column of location probabilities for the data used to train the model
    assumed that the data is in chronological order with row 0 being oldest case

    Parameters
    ----------
    data : an matrix of previous sighting location infomation- given in format
            [latitude, longitude] including the sightings in prior
    prior : an nx2 matrix of n previous locations, default is location of the first positive 
            case in the data on 9/19/2019
    distance : whatever function is being used to calculate the distance

    Returns
    -------
    a vector of the location probabilities for the data 

    '''
    
    location_probabilities = np.ones(len(prior)) # We know that the prior known cases were true
    k = len(prior)
    for i in range(k, len(data)): # for each row in the data past priors
        # Calculate the location_probability and add to our vector
        p = location_probability(prior, data[i,:], distance)
        
        #Add this probability to the vector
        location_probabilities = np.append(location_probabilities, p)
        
        # add the location to priors if it was actually positive
        if data[i, 0] == 1: # 1 means 'Positive ID', we converted these already
            
            prior = np.vstack([prior, data[i, 1:]])
        # repeat!
    
    return location_probabilities


