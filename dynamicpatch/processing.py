# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 16:27:36 2024
This script generates outputs of PATCHES by calling other functions and create
maps and graphics, the main function mainly calls this script to get the ouputs


@author: Aiyin Zhang
"""
from osgeo import gdal
import numpy as np
import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
#import identify_dypatch
from dynamicpatch import TransitionAnalysis,config,WriteData
import glob

from dynamicpatch.config import in_params, proc_params, data_val, res
workpath, year, connectivity, targ_pre, in_nodata, FileType, dataset,study_area = in_params['workpath'],\
    in_params['years'],in_params['connectivity'],in_params['presence'], in_params['nodata'],\
        in_params['FileType'], in_params['dataset'],in_params['study_area']

params = proc_params
absence, presence, nodata, nt, nl, ns, connectivity = params

## need to include res  
if (FileType == 'Tif' or FileType == 'Folder'):
    from dynamicpatch.config import data 
#%% 
def initialize():
    map_figs = []
    from dynamicpatch import create_maps
    for i in range(len(year)):
        
        binary_t = np.zeros((nl,ns),dtype = 'ubyte')
        binary_t[data_val[i,:,:] == targ_pre] = presence
        binary_t[data_val[i,:,:] != targ_pre] = absence
        binary_t[data_val[i,:,:] == in_nodata] = nodata         
        
        map_fig = create_maps.map_timepoint(i,binary_t,absence,presence,res = res)
        map_figs.append(map_fig)
    return map_figs, binary_t


def run_analysis(mapshow = True, chartsshow = True,unit = None, data = None, export_map = False, progress = None, width = 0.35, log_scale = True):           
    is_complete = False

    pattern = np.zeros((nt,nl,ns),dtype = 'int')
    
    pattern_maps = []
    generated_charts = []
    chart_titles = []
    analysis = {}
    df_inde_all = []
    map_title = ''
    for i in range(nt):
        
        binary = np.zeros((2,nl,ns),dtype = 'ubyte')

        binary[data_val[i:i+2,:,:] == targ_pre] = presence
        binary[data_val[i:i+2,:,:] != targ_pre] = absence
        binary[data_val[i:i+2,:,:] == in_nodata] = nodata
            
        analysis[i] = TransitionAnalysis.TransitionAnalysis(params, binary[0], binary[1], year)
        pattern[i] = analysis[i].identify()
        
            
    if mapshow is True:
        from dynamicpatch import create_maps
        map_title = f'Transition Pattern at {study_area}'
        for i in range(nt):
            pattern_map = create_maps.pattern_map(i,pattern,data,res)  
            pattern_maps.append(pattern_map)
            
    if export_map is True:                     
        # Prompt user for output directory
        output_dir = input("Enter output map directory (please end with / or \\): ").strip()
        
        # Ensure the directory ends with a backslash or forward slash
        if not output_dir.endswith('\\') and not output_dir.endswith('/'):
            output_dir += '\\'  # Use '\\' for Windows paths, '/' for Unix-like paths
    
        # Create the full path to the file
        FileName = output_dir + dataset + '_trans_type.tif'
        print(FileName,data,pattern)
        # Call the function with the new FileName
        WriteData.writedata(FileName, pattern, data, 'byte')
        
    if chartsshow is True: 
        from dynamicpatch import create_charts
        df_inde_all=pd.DataFrame(columns = ['year','Disappearing','Appearing','Splitting','Merging'])   
        df_inde_all['year']=year[0:-1]
        for i in range(nt):
            df_inde_all.iloc[i,1:] = analysis[i].gross_change()
        show_charts = create_charts.Gen_Charts(pattern,areaunit = unit)
        fig1,title1 = show_charts.plot_ave_size(width = width, log_scale = log_scale)
        fig2,title2 = show_charts.plot_num(width = width)
        
        fig3, title3 = show_charts.gainloss_stackedbars()
        fig4, title4 = show_charts.inde_stackedbars(df_inde_all)
        
        generated_charts.extend([fig1, fig2, fig3, fig4])
        chart_titles.extend([title1,title2,title3,title4])
    
    is_complete = True 
    outputs = df_inde_all,data,data_val,binary
    result = pattern, pattern_maps, map_title, generated_charts, chart_titles, outputs
    return result

def write_image(pattern,FileName):
    from dynamicpatch import WriteData
    WriteData.writedata(FileName, pattern,data,'byte')
    

