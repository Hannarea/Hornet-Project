# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 14:08:02 2021

@author: hreed
"""

# This is intended to be used as a module for reading in data from the excel spreadsheet of data

import pandas as pd
import numpy as np

def get_data_col(file_path, columns):
    '''
    gets the specified data columns

    Parameters
    ----------
    file_path : use format r'file path\file_name.xlsx'
    columns : list of strings for the columns you are retrieving

    Returns
    -------
    df : specified column or columns of data

    '''
    data = pd.read_excel(file_path)
    df = pd.DataFrame(data, columns = columns)
    return df
    

# Example of how to use:
file_path = r'C:\Users\hreed\Documents\UCF\Projects\Hornet-Project\TestData.xlsx'
a = get_data_col(file_path, ['Size', 'Color', 'Binary'])

def convert_to_array(data):
    '''
    

    Parameters
    ----------
    data : a matrix of data, assumed to be integer valued 

    Returns
    -------
    the data converted to an np.array

    '''
    return np.array(data)

b = convert_to_array(a)
print(b)

