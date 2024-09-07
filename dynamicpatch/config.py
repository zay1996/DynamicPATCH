# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 16:59:49 2024
This script contains all the important parameters used in DynamicPATCH

@author: Aiyin Zhang
"""

import tkinter as tk
from tkinter import Tk
import pandas as pd 
from dynamicpatch import read_data 


cat_list=['No Data','Stable Absence','Appearing','Merging','Filling','Expanding',\
          'Disappearing','Splitting','Perforating','Contracting','Stable Presence']

cat_ind = [-1,0,1,2,3,4,5,6,7,8,9]
cat_dict = dict(zip(cat_list, cat_ind))


df_cat = pd.DataFrame(columns = ['Value','Type','Color'])
df_cat['Type'] = cat_list
df_cat['Value'] = cat_ind
df_cat.loc[df_cat['Type'] == 'No Data', 'Color'] = 'white'
df_cat.loc[df_cat['Type'] == 'Stable Absence', 'Color'] = 'silver'
df_cat.loc[df_cat['Type'] == 'Disappearing', 'Color'] = '#993404'
df_cat.loc[df_cat['Type'] == 'Splitting', 'Color'] = 'crimson'
df_cat.loc[df_cat['Type'] == 'Contracting', 'Color'] = 'yellow'
df_cat.loc[df_cat['Type'] == 'Perforating', 'Color'] = 'orange'
df_cat.loc[df_cat['Type'] == 'Appearing', 'Color'] = '#0570b0'
df_cat.loc[df_cat['Type'] == 'Merging', 'Color'] = '#00A9E6'
df_cat.loc[df_cat['Type'] == 'Expanding', 'Color'] = '#807dba'
df_cat.loc[df_cat['Type'] == 'Filling', 'Color'] = 'aqua'
df_cat.loc[df_cat['Type'] == 'Stable Presence', 'Color'] = 'grey'


# Global variables to store parameters
in_params = {}
proc_params = None
data = None
data_val = None
workpath = ""
targ_pre = 0
in_nodata = 0
connectivity = 0
year = []
filetype = ""
res = 0
nt = 0
study_area = ""
dataset = ""
params = None

def read_params_interface():
    '''
    Read parameters by pop up window 

    Returns
    -------
    None.

    '''
    global in_params, proc_params, data, data_val
    global workpath, targ_pre, in_nodata, connectivity, year, filetype,res, nt
    global study_area, dataset

    from dynamicpatch.interface import InputApp, process_inputs
    
    
    # Create the Tkinter application
    root = Tk()  # Create a root Tk instance to manage the Tkinter window
    root.withdraw()  # Hide the root window (we only want the App window to show)

    app = InputApp()
    app.mainloop()  # This call will block until the window is closed
    in_params = app.get_params()
    if in_params:
        process_inputs(**in_params)
    
    
    workpath = in_params['workpath']
    targ_pre = int(in_params['presence'])
    in_nodata = int(in_params['nodata'])
    connectivity = int(in_params['connectivity'])
    year = in_params['years']
    filetype = in_params['FileType']
    dataset = in_params['dataset']
    study_area = in_params['study_area']
    
    ### READ ALL TIF FILES UNDER WORKPATH 
    if (filetype == 'Tif' or filetype == 'Folder'):
        data, data_val,size = read_data.readdatafunc(filetype, workpath)
        res = round(data.GetGeoTransform()[1])
    else:
        data_val,size = read_data.readdatafunc(filetype,workpath)
        
    nl,ns = size[-2:]   
    nt = len(year)-1
        
    presence = 2
    absence = 1
    nodata = 0 
    
    
    proc_params = absence, presence, nodata, nt, nl, ns, connectivity  

    
    return proc_params, data, data_val
    

def read_params(workpath,year,targ_pre, connectivity = 8, in_nodata = 0):
    '''
    Read parameters option 1: specifying dataset 
    

    Parameters
    ----------
    dataset : TYPE
        DESCRIPTION.
    type_ : TYPE, optional
        DESCRIPTION. The default is 'change'.

    Returns
    -------
    params : TYPE
        DESCRIPTION.

    '''
    global params

    workpath, datatype,targ_pre,in_nodata,year,connectivity,res = read_data.readparams(dataset,type_='compare')


    ### READ ALL TIF FILES UNDER WORKPATH 
    if (datatype == 'Tif'):
        data, data_val,size = read_data.readdatafunc(datatype, workpath)
    else:
        data_val,size = read_data.readdatafunc(datatype,workpath)
        
    nl,ns = size[-2:]
    nt = len(year)-1
        
    presence = 2
    absence = 1
    nodata = 0 
    
    
    absence, presence, nodata, nt, nl, ns, connectivity = params
    
    return params 

