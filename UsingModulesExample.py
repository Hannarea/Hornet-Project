# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 00:49:05 2021

@author: hreed
"""

# This is an example of how to use modules for our project

# You need to import the functions from the file (this is assuming the file is in the smae folder)
from readData import  get_data_col, convert_to_array, convert_binary, select_data, positive_locations
from LocationProbability import build_location_probabilities_vector

# Define the filepath for the data
file_path = r'C:\Users\hreed\Documents\UCF\Projects\Hornet-Project\2021MCMProblemC_DataSet.xlsx'

# retrieve the selected columns
data = convert_to_array(get_data_col(file_path, ['Lab Status', 'Latitude', 'Longitude']))

# kick out the unwanted data
SubData = select_data(data, 0, ['Positive ID', 'Negative ID'])

# converts the positive ID to 1 and other to 0
final = convert_binary(SubData, 0, 'Positive ID')

location_probabilities = build_location_probabilities_vector(final)
print(location_probabilities)


# This is an array with the positive locations 
L = positive_locations(final, 0, 1, 1)
print(L)
