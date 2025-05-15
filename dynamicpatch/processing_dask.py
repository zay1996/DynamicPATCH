# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 16:27:36 2024
This script generates outputs of PATCHES by calling other functions and create
maps and graphics, the main function mainly calls this script to get the ouputs


@author: Aiyin Zhang
"""

import numpy as np
import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
#import identify_dypatch
from dynamicpatch import TransitionAnalysis,config,WriteData
import glob
import dask.array as da 

#%% 
def initialize(params,data_val):
    map_figs = []
    from dynamicpatch import create_maps
    targ_pre,in_nodata = params['presence'],params['nodata']
    presence,absence,nodata = params['proc_presence'],params['proc_absence'],params['proc_nodata']
    res = params['res']
    year = params['years']
    nl = params['nl']
    ns = params['ns']
    for i in range(len(year)):
        
        binary_t = np.zeros((nl,ns),dtype = 'ubyte')
        binary_t[data_val[i,:,:] == targ_pre] = presence
        binary_t[data_val[i,:,:] != targ_pre] = absence
        binary_t[data_val[i,:,:] == in_nodata] = nodata         
        
        map_fig = create_maps.map_timepoint(year,i,binary_t,absence,presence,res = res)
        map_figs.append(map_fig)
    return map_figs, binary_t

def reclass(input_,pres_val,nodata_val,chunking = None):
    
    reclassed = da.full_like(input_, fill_value=1, dtype="ubyte")

    # Apply conditions using Dask's `where()` (NumPy-compatible)
    reclassed = da.where(input_ == pres_val, 2, reclassed)  # Set 2 where combined_data == 3
    reclassed = da.where(input_ == nodata_val, 0, reclassed)  # Set 0 where combined_data == 0
    

    return reclassed

def run_analysis(params,
                 data_val,
                 mapshow = True, 
                 chartsshow = True,
                 unit = None, 
                 export_map = False, 
                 progress = None, 
                 width = 0.35, 
                 log_scale = True,
                 rotation = 0,
                 chunk_size = 1000):           
    is_complete = False

    workpath, year, connectivity, targ_pre, in_nodata, FileType, dataset,study_area = \
        params['workpath'], params['years'],params['connectivity'],params['presence'], params['nodata'],\
            params['FileType'], params['dataset'],params['study_area']
    
    nt,nl,ns = params['nt'],params['nl'],params['ns']

    presence,absence,nodata = params['proc_presence'],params['proc_absence'],params['proc_nodata']

    res = params['res']

    proc_params = absence, presence, nodata, nt, nl, ns, connectivity  
    pattern = da.zeros((nt,nl,ns),dtype = 'int')
    
    pattern_maps = []
    generated_charts = []
    chart_titles = []
    analysis = {}
    df_inde_all = []
    map_title = ''


    for i in range(nt):
        chunking = (nt, chunk_size,chunk_size)
        binary = reclass(data_val[i:i+2].data, presence, nodata, chunking = chunking)
        
            
        analysis[i] = TransitionAnalysis.TransitionAnalysis(proc_params, binary[0], binary[1], year)
        pattern[i] = analysis[i].identify()
        
            
    if mapshow is True:
        from dynamicpatch import create_maps
        map_title = f'Transition Pattern at {study_area}'
        for i in range(nt):
            pattern_map = create_maps.pattern_map(year,i,pattern,res = res)  
            pattern_maps.append(pattern_map)
            
    if export_map is True:                     
        # Prompt user for output directory
        output_dir = input("Enter output map directory (please end with / or \\): ").strip()
        
        # Ensure the directory ends with a backslash or forward slash
        if not output_dir.endswith('\\') and not output_dir.endswith('/'):
            output_dir += '\\'  # Use '\\' for Windows paths, '/' for Unix-like paths
    
        # Create the full path to the file
        FileName = output_dir + dataset + '_trans_type.tif'
        print(FileName,pattern)
        # Call the function with the new FileName
        WriteData.writedata_rio(FileName, pattern)
        
    if chartsshow is True: 
        from dynamicpatch import create_charts
        df_inde_all=pd.DataFrame(columns = ['year','Disappearing','Appearing','Splitting','Merging'])   
        df_inde_all['year']=year[0:-1]
        for i in range(nt):
            df_inde_all.iloc[i,1:] = analysis[i].gross_change()
        show_charts = create_charts.Gen_Charts(pattern,year,connectivity, nt, res,areaunit = unit)
        fig1,title1 = show_charts.plot_ave_size(width = width, log_scale = log_scale)
        fig2,title2 = show_charts.plot_num(width = width)
        
        fig3, title3 = show_charts.gainloss_stackedbars(rotation = rotation)
        fig4, title4 = show_charts.inde_stackedbars(df_inde_all,rotation = rotation)
        
        generated_charts.extend([fig1, fig2, fig3, fig4])
        chart_titles.extend([title1,title2,title3,title4])
    
    is_complete = True 
    outputs = df_inde_all,data_val,binary
    result = pattern, pattern_maps, map_title, generated_charts, chart_titles, outputs
    return result

def write_image(pattern,data,FileName):
    from dynamicpatch import WriteData
    WriteData.writedata(FileName, pattern,data,'byte')
    

