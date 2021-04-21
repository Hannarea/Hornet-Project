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
print(a)
print(b)
print(a.shape)
print(b.shape)


def convert_binary(data, col, true):
    '''
    Converts data entries with binary answers in string format to 0's and 1's

    Parameters
    ----------
    data : the matrix of elements
    col : The column we are converting to 0's and 1's (first column is 0)
    true : the string that is to be converted into a 1. All others will become a 0

    Returns
    -------
    the altered matrix

    '''
    
    for i in range(len(data)):
        if (data[i,col] == true):
            data[i,col] = 1
        else:
            data[i,col] = 0
            
    return data

c = convert_binary(b, 2, 'yes')
print(c)

def select_data(data, col, keep):
    '''
    reduces the data besed on the 'keep' values in the specified column

    Parameters
    ----------
    data : matix of original data
    col : the column to look at during the selection process, first column is 0
    keep : the values in the column to keep, array type

    Returns
    -------
    new matrix

    '''
    i=0
    while (i < len(data)):
        if data[i,col] not in keep:
            data = np.delete(data, i, 0)
            print('deleting row', i)
        i+=1
    return data

keep = ['red', 'blue']
d = select_data(c, 1, keep)
print(d)