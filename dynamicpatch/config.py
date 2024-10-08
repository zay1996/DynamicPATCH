# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 16:59:49 2024
This script contains all the important parameters used in DynamicPATCH

@author: Aiyin Zhang
"""

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


def read_params(_workpath,_year,_targ_pre = 1, _connectivity = 8, _in_nodata = 0, _study_area = None):
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
    global in_params, proc_params, data, data_val
    global workpath, targ_pre, in_nodata, connectivity, year, filetype,res, nt
    global study_area, dataset
    
    workpath,year,targ_pre,connectivity,in_nodata,study_area = \
    _workpath,_year,_targ_pre,_connectivity,_in_nodata,_study_area
        
    filetype,dataset = read_data.check_filetype(workpath)
    
    if study_area is not None:
        dataset = study_area
        

    ### READ ALL TIF FILES UNDER WORKPATH 
    if (filetype == 'Tif' or filetype == 'Folder'):
        data, data_val,size = read_data.readdatafunc(filetype, workpath)
        res = round(data.GetGeoTransform()[1])
    else:
        data_val,size = read_data.readdatafunc(filetype,workpath)
        res = 0
 
    in_params = {
        'workpath': workpath,
        'years': year,
        'connectivity': connectivity,
        'presence': targ_pre,
        'nodata': in_nodata,
        'FileType': filetype,
        'dataset': dataset, 
        'study_area':study_area
    }        
 
    
    nl,ns = size[-2:]   
    nt = len(year)-1
        
    presence = 2
    absence = 1
    nodata = 0 
    
    
    proc_params = absence, presence, nodata, nt, nl, ns, connectivity  
    
    return proc_params, data, data_val 

