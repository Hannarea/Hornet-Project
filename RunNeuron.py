# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 10:52:08 2021

@author: hreed
"""

import numpy as np
from SimpleNetwork import OurNeuralNetwork

# Read in the data to train the model 

# Convert the data to the proper format (an nx3 matrix): 
#       [location_probability, Color, Abdomin]
# Get the 'answers' for the test data: 
#       [Lab Status]



# For now, some fake data and answers: 
data = np.array([
    [0,0,0],
    [1,0,0],
    [0,1,0],
    [1,1,1]
    ])

lab_status = np.array([0, 0, 1, 1])

# Train the model 

network = OurNeuralNetwork()
network.train(data, lab_status)


# Run some test cases

x1 = np.array([0, 0, 0]) # 128 pounds, 63 inches
x2 = np.array([1, 1, 1])  # 155 pounds, 68 inches
print("x1: %.3f" % network.feedforward(x1)) # 0.951 - F
print("x2: %.3f" % network.feedforward(x2)) # 0.039 - M