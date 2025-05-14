# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 15:59:21 2024

@author: AiZhang
"""
import pandas as pd
import numpy as np
import os
import re 
import glob
import rioxarray 
import xarray as xr
import dask 
import rasterio 
from pathlib import Path
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
        if (FileType == 'Tif' and FilePath.lower().endswith('.tif')):
            with rasterio.open(FilePath) as src:
                data = src.read()  
                size = data.shape

        # if workpath indicates a folder 
        elif isinstance(FilePath, (str, Path)) and os.path.isdir(FilePath):
            tif_files = sorted(glob.glob(f"{FilePath}/**/*.tif", recursive=True))
            if(len(tif_files) != len(tif_files)):
                raise ValueError("Number of files does not match number of time points!")  
            datasets = []
            for i,f in enumerate(tif_files):
                with rasterio.open(f) as src:
                    data = src.read()  # shape: (bands, height, width)
                    datasets.append(data)

            # Check shape consistency
            first_shape = datasets[0].shape
            if not all(d.shape == first_shape for d in datasets):
                raise ValueError("All rasters must have the same shape and band count.")

            # Stack into a NumPy array: shape (time, bands, height, width)
            data = np.stack(datasets, axis=0)
            size = data.shape

        elif(type(FilePath)==list):
            datasets = []
            for i in FilePath:
                with rasterio.open(i) as src:
                    data = src.read()[0]  # shape: (bands, height, width)
                    datasets.append(data)
            data = np.stack(datasets, axis=0)
            if(len(np.shape(data))==3):
                size = np.shape(data)[1:]
            elif(len(np.shape(data))==2):
                size = np.shape(data)

        return src,data,size


def readdatafunc_new(FileType, FilePath,chunk_size = 5000):
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
        # if workpath indicates a tif file
        if (FilePath.lower().endswith('.tif')):
            data = rioxarray.open_rasterio(FilePath).chunk({"band": -1, "y": chunk_size, "x": chunk_size})
            size = data.shape

        # if workpath indicates a folder 
        elif(os.path.isdir(FilePath)):
            tif_files = sorted(glob.glob(f"{FilePath}/**/*.tif", recursive=True))
            if(len(tif_files) != len(tif_files)):
                raise ValueError("Number of files does not match number of time points!")  
            datasets = []
            for i,f in enumerate(tif_files):
                # update filenname if needed 
                tif = tif_files[i]
                map_ = tif
                raster_map = rioxarray.open_rasterio(map_).chunk({"band": -1, "y": chunk_size, "x": chunk_size})
                raster_map = raster_map.assign_coords(band=[f])
                #raster_map.attrs["long_name"] = f"classification_{year}"
                datasets.append(raster_map)

            data = xr.concat(datasets, dim='band')
            size = data.shape

        return data,size


def check_filetype(file_path):
    # Determine the file type based on the extension
    if os.path.isdir(file_path):
        file_type = "Folder"
        dataset = os.path.basename(file_path)
    else:
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        if ext == ".tif":
            file_type = "Tif"
        elif ext == ".xlsx":
            file_type = "Excel"
        elif ext == ".csv":
            file_type = "Csv"
        else:
            print("Input Error", "The input file must be of type .tif, .xlsx, or .csv.")
            file_type = None
            
        # Extract name of the dataset
        dataset = os.path.basename(file_path).split('.')[0]
    return file_type, dataset
