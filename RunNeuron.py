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

# get the input data to tring the network [location_probability, Color, Abdomin]
training_data = np.hstack((location_probabilities, SubData[:, 3], SubData[:, 4]))
training_data = (training_data.reshape((3,798))).T
##############################################################################






# ---------------
# Train the model
# --------------- 
###############################################################################

network = OurNeuralNetwork()
network.train_bce(training_data, lab_status)

# Here are the resulting weights put on the location_prob, color, abdomin
print('[location_prob, color, abdomin]')
print(network.w)
print('Bias:\t', network.b)

###############################################################################



# --------------------
# Run some test cases
# --------------------
################################################################################

# network = OurNeuralNetwork()
# network.b = -5.78491211151786
# network.w = np.array([-1.12998904, -1.35090052, -0.36456844])

test1 = np.array([1, 1, 1])
ans = network.feedforward(test1)
print("we want near 1", ans)

test2 = np.array([0, 0, 0])
print("We want near 0", network.feedforward(test2))

################################################################################