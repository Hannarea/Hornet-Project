# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 17:00:02 2021

@author: hreed
"""

import numpy as np
import math
import matplotlib.pyplot as plt

# This is intended to be used as a module from which location_probability can be called. 

# To do: write code to plt the coordinates on 2D grid




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

#### Testing the arc_length function ###
# The distance between Big Ben in London (51.5007° N, 0.1246° W) and 
# The Statue of Liberty in
# New York (40.6892° N, 74.0445° W) is 5574.8 km. 

# a = np.array([51.5007, 0.1246])
# b = np.array([40.6892, 74.0445])
# print(arc_length(a, b))


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
    if x>30:
        return 1
    return 0
    # sig = 100
    # return 1- np.exp(-1/2*(x/sig)**2)

    

# Euclidean Norm if using rectangular coordinates for locations 
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
    
    #Hello Hanna
    
    location_probabilities = np.ones(len(prior)) # We know that the prior known cases were true
    k = len(prior)
    for i in range(k, len(data)): # for each row in the data
        # Calculate the location_probability and add to our vector
        p = location_probability(prior, data[i,:], distance)
        location_probabilities = np.append(location_probabilities, p)
        # add the location to priors if it was actually positive
        if data[i, 0] == 1: # 1 means 'Positive ID', we converted these already
            prior = np.vstack([prior, data[i, 1:]])
        # repeat!
    return location_probabilities



# # Example - the red dots have high probability and the blue have low probability        
# prior = np.array([[0,0]])
# data = np.array([
#     [0, 0, 0],
#     [0, 0, 1],
#     [0, -.5, .5],
#     [0, .5, -.5],
#     [0, -1, 0],
#     [0, -.2, 0],
#     [0, .2, 0],
#     [0, 1, 0],
#     [0, -.5, -.5],
#     [0, .5, -.5],
#     [0, 0, -1],
#     [0, 0, .2],
#     [0, 0, -.2],
#     [1, .5, .5],
#     [0, .7, .6], # Here, we have reds because of the true case right before
#     [0, .6, .4],
#     [0, .4, .55]
#     ])
# # norm and arc_length can be exchanged. 
# v = build_location_probabilities_vector(data, norm, prior = prior)
# print(v)

# # the small dots show where the points like on the pdf. they should decrese as the points get farther from the origin in this example
# for i in range(len(data)):
#     if v[i] > 0.5:
#         plt.plot(data[i,1], data[i,2], 'ro')
#     else:
#         plt.plot(data[i,1], data[i,2], 'bo')

    

    