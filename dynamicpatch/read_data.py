# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 15:59:21 2024

@author: AiZhang
"""
import pandas as pd
import numpy as np
from osgeo import gdal
import os
import re 
import glob

def readdatafunc(FileType, FilePath):
    '''
    Function of reading data, now support reading .tif, .csv, and .xlsx.
    Note that the tabular input data (.csv and .xlsx) need to be constructed
    in the format of a 2 dimensional matrix. 

    Parameters
    ----------
    FileType : String
        Type of the file, now support 4 options: 'Tif','Csv','Excel','Folder'. 
        If the file type is folder, the files within the folder need to be tif files. 
    FilePath : String
        Directory to where the data is stored

    Returns
    -------
    data: gdal object
        Gdal dataset
    data_val: NumPy array 
        The input data in the form of a NumPy array
    size: tuple
        Size of the data
    '''
    year_pattern = re.compile(r'(\d+)\.tif$')
    if(FileType == 'Csv'):
        if(type(FilePath)==str):
            with open(FilePath, 'r', encoding='utf-8-sig') as f: 
                data = np.genfromtxt(f, dtype=float, delimiter=',')
            data = data.astype('byte')
            size = np.shape(data)
        if(type(FilePath)==list):
            data = {}
            dataar = []
            for i in range(len(FilePath)):
                with open(FilePath[i], 'r', encoding='utf-8-sig') as f: 
                    data[i] = np.genfromtxt(f, dtype=float, delimiter=',')
                    dataar.append(data[i])
                    
                    data[i] = data[i].astype('byte')
            size = np.shape(data)[-2:]
        return data,size
    if(FileType == 'Excel'):
        # Load all sheets into a dictionary of DataFrames
        sheets = pd.read_excel(FilePath, sheet_name=None, header = None)
        
        # Convert the DataFrames to NumPy arrays and collect them in a list
        arrays_list = [df.to_numpy() for df in sheets.values()]
        
        # Find the maximum shape in each dimension
        max_rows = max(array.shape[0] for array in arrays_list)
        max_cols = max(array.shape[1] for array in arrays_list)
        
        # Initialize a 3D NumPy array with the maximum shape and fill with NaNs
        num_sheets = len(arrays_list)
        data = np.full((num_sheets, max_rows, max_cols), np.nan,dtype='ubyte')
        
        # Copy the values from the 2D arrays into the 3D array
        for i, array in enumerate(arrays_list):
            rows, cols = array.shape
            data[i, :rows, :cols] = array
        size = np.shape(data)[-2:]
        
        return data,size

    if(FileType == 'Tif' or FileType == 'Folder'):
        if(type(FilePath)==list):
            data = {}
            data_val = []
            for i in range(len(FilePath)):
                data[i] = gdal.Open(FilePath[i])
                data_val.append(data[i].ReadAsArray().astype('byte'))
            data_val = np.array(data_val)
            if(len(np.shape(data_val))==3):
                size = np.shape(data_val)[1:]
            elif(len(np.shape(data_val))==2):
                size = np.shape(data_val)
        # check if its a folder or if its a tif file
        elif (os.path.isdir(FilePath)): ## if its a folder          
            tif_files = glob.glob(os.path.join(FilePath, '*.tif'))
            data_gdal = {}
            data_dict = {}
            for i,f in enumerate(tif_files):
                match = year_pattern.search(f)
                if match:
                    number = match.group(1)
                    var_name = f'{number}'
                data_gdal[i] = gdal.Open(tif_files[i])
                data_dict[i] = data_gdal[i].ReadAsArray().astype('byte')
                size = np.shape(data_dict[i])
            data_val = np.stack(list(data_dict.values()))
            data = data_gdal[0]
            #data_val = data_dict
        elif os.path.isfile(FilePath):
            data = {}
            data = gdal.Open(FilePath)
            data_val = data.ReadAsArray().astype('byte')
            if(len(np.shape(data_val))==3):
                size = np.shape(data_val)[1:]
            elif(len(np.shape(data_val))==2):
                size = np.shape(data_val)
        return data,data_val,size



