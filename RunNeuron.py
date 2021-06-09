# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 10:52:08 2021

@author: hreed
"""

import numpy as np
from SimpleNetwork import OurNeuralNetwork
from readData import convert_to_array, get_data_col
from LocationProbability import build_location_probabilities_vector, arc_length

# -------------------------------------------------------------
# Read in the data to train the model with the required columns
# -------------------------------------------------------------
##########################################################################
# [Lab Status, Latitude, Longitude, Color, Abdomin]

# Define the filepath for the data
file_path = r'C:\Users\hreed\Documents\UCF\Projects\Hornet-Project\HornetNewDataWithPositiveCases.xlsx'

# retrieve the selected columns
data = convert_to_array(get_data_col(file_path, ['Lab Status', 'Latitude', 'Longitude', 'Color', 'Abdomen']))

SubData = data

#############################################################################


# -----------------------------------------------------
# Convert the data to the proper format (an nx3 matrix): 
#       [location_probability, Color, Abdomin]
# Get the 'answers' for the test data: 
#       [Lab Status]
# -----------------------------------------------------
#############################################################################

# Makes the lab_status vector with 0's and 1's
lab_status = SubData[:, 0]
for i in range(len(lab_status)):
    if lab_status[i] == 'Positive ID':
        lab_status[i] = 1
    else:
        lab_status[i] = 0


# Build the location_probabilities based on the latitude and longitude
locations = SubData[:,1:3]
location_probabilities = build_location_probabilities_vector(locations, arc_length)


# Writing the location probabilities to excel sheet 
import xlsxwriter
workbook = xlsxwriter.Workbook('location_probabilities_vector.xlsx')
worksheet = workbook.add_worksheet()
lst = []
for i in range(len(location_probabilities)):
    lst.append(location_probabilities[i])

worksheet.write_column(0,0,lst)
workbook.close()


# get the input data to tring the network [location_probability, Color, Abdomin]
training_data = np.hstack((location_probabilities, SubData[:, 3], SubData[:, 4]))
training_data = (training_data.reshape((3,775))).T
##############################################################################

# ---------------
# Train the model
# --------------- 
###############################################################################

network = OurNeuralNetwork()
network.train_gradient_descent(training_data, lab_status)

# Here are the resulting weights put on the location_prob, color, abdomin
print('[location_prob, color, abdomin]')
print(network.w)
print('Bias:\t', network.b)

###############################################################################


# --------------------
# Run some test cases
# --------------------
################################################################################

# Generates 100 random sightings
from random import random 

test_cases = np.zeros((1000, 3))
for i in range(1000):
    test_cases[i, 0] = random()
    if (random()>0.5):
        test_cases[i,1] = 1
    if (random()>0.5):
        test_cases[i,2] = 1

results = []
for i in range(1000):
    results.append(network.feedforward(test_cases[i]))
    
s = 0
for i in range(1000):
    if (test_cases[i,0]>0.75) and (test_cases[i,1] == test_cases[i,2] == 1):
        s += 1
print(s)


for i in range(1000):
    if results[i]>0.5:
        print(results[i], i)

