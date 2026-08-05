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
import math 
import matplotlib.patches as mpatches
from dynamicpatch.config_new import df_cat
def initialize(params,data_val,show_map = False):
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
        if(show_map is True):
            map_fig = create_maps.map_timepoint(year,i,binary_t,absence,presence,res = res)
            map_figs.append(map_fig)
    return map_figs, binary_t

def initialize_tk():
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

def run_analysis(params,
                 data,
                 data_val,
                 mask = None,
                 mapshow = True, 
                 chartsshow = True,
                 unit = None, 
                 export_map = None,
                 gif_map = False, 
                 progress = None, 
                 width = None, 
                 log_scale = True,
                 rotation = 0,
                 figsize = (12,6),
                 legend = 'right'):           
    is_complete = False

    workpath, year, connectivity, targ_pre, in_nodata, FileType, dataset,study_area = \
        params['workpath'], params['years'],params['connectivity'],params['presence'], params['nodata'],\
            params['FileType'], params['dataset'],params['study_area']
    
    nt,nl,ns = params['nt'],params['nl'],params['ns']

    presence,absence,nodata = params['proc_presence'],params['proc_absence'],params['proc_nodata']

    res = params['res']

    proc_params = absence, presence, nodata, nt, nl, ns, connectivity  
    pattern = np.zeros((nt,nl,ns),dtype = 'int')
    
    pattern_maps = []
    generated_charts = []
    chart_titles = []
    analysis = {}
    df_inde_all = []
    map_title = ''


    for i in range(nt):
        print("Running Transition Analysis on time interval",i)    
        binary = np.zeros((2,nl,ns),dtype = 'ubyte')

        binary[data_val[i:i+2,:,:] == targ_pre] = presence
        binary[data_val[i:i+2,:,:] != targ_pre] = absence
        binary[data_val[i:i+2,:,:] == in_nodata] = nodata
            
        analysis[i] = TransitionAnalysis.TransitionAnalysis(proc_params, binary[0], binary[1], year)
        pattern[i] = analysis[i].identify()
        
            
    if mapshow is True:
        from dynamicpatch import create_maps
        categorylist = list(df_cat.sort_values(by='Value')['Type'])
        colorlist = [df_cat.loc[df_cat['Type'] == cat, 'Color'].values[0] for cat in categorylist]

        if(nt<=5):
            map_title = f'Transition Pattern at {study_area}'
            for i in range(nt):
                pattern_map = create_maps.pattern_map(year,i,pattern,res = res)  
                pattern_maps.append(pattern_map)
            
        if(nt > 5):

            # --- 1️⃣ Dynamically determine layout ---
            # Try to make it more landscape: more columns than rows
            ncols = math.ceil(math.sqrt(nt))
            nrows = math.ceil(nt / ncols)
            aspect = nl / ns

            subplot_width = 5    # Inches per subplot (adjustable)
            subplot_height = subplot_width / aspect

            fig_width = subplot_width * ncols
            fig_height = subplot_height * nrows + 1.5    # extra space for legend

            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(fig_width, fig_height),
                constrained_layout=True
            )

            #print("update?")
            axes = np.array(axes).reshape(-1)
            for i, ax in enumerate(axes):
                if i < nt-1:
                    im = create_maps.pattern_map(year,i,pattern,res = res,ax = ax,north_arrow = False)  
                    ax.set_title(str(year[i])+'-' +str(year[i+1]), fontsize=14, pad=8)
                    ax.axis('off')
                if i == nt-1:
                    im = create_maps.pattern_map(year,i,pattern,res = res,ax = ax,north_arrow = True)  
                    ax.set_title(str(year[i])+'-' +str(year[i+1]),fontsize=14, pad=8)
                    ax.axis('off')
                else:
                    # Hide any extra subplot if grid > number of maps
                    ax.axis('off')

            patches = [mpatches.Patch(color=colorlist[i], label=categorylist[i]) for i in range(len(categorylist))][1:]
            fig.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol = 6,fontsize=20)

    if gif_map is True:
        import io
        import imageio.v2 as imageio  # <-- use v2 explicitly
        # Loop through time steps and generate figures
        frames = []
        for i in range(nt-1):
            fig = create_maps.pattern_map(year, i, pattern, data, res)

            # Save the figure to a buffer instead of disk
            # Save to buffer
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)

            image = imageio.imread(buf)  # <-- v2-compatible read
            frames.append(image)

            buf.close()
            plt.close(fig)

        # Save as GIF (2 seconds per frame => 0.5 FPS)
        #imageio.mimsave('transition
        imageio.mimsave('transition_pattern'+study_area+'.gif', frames, fps = 1, loop = 0) 


    if export_map is not None:
        output_dir = export_map                     
    
        # Create the full path to the file
        FileName = output_dir + '.tif'
        print(FileName,data,pattern)
        # Call the function with the new FileName
        WriteData.writedata_rasterio(FileName, data,pattern)
        
    if chartsshow is True: 
        from dynamicpatch import create_charts
        df_inde_all=pd.DataFrame(columns = ['year','Disappearing','Appearing','Splitting','Merging'])   
        df_inde_all['year']=year[0:-1]
        for i in range(nt):
            df_inde_all.iloc[i,1:] = analysis[i].gross_change()
        show_charts = create_charts.Gen_Charts(pattern,year,connectivity, nt, res,mask = mask, areaunit = unit)

        df_patch_size,fig1,title1 = show_charts.plot_ave_size(width = width, log_scale = log_scale)
        df_patch_num,fig2,title2 = show_charts.plot_num(width = width)
        
        fig3, title3,df_gainloss_all = show_charts.gainloss_stackedbars(rotation = rotation,legend = legend, figsize = figsize)
        fig4, title4 = show_charts.inde_stackedbars(df_inde_all,rotation = rotation)
        
        generated_charts.extend([fig1, fig2, fig3, fig4])
        chart_titles.extend([title1,title2,title3,title4])
    
    is_complete = True 
    #outputs = df_inde_all,df_gainloss_all,df_patch_size,df_patch_num,data,data_val,binary
    outputs = {
    "pattern": pattern,
    "patch_size": df_patch_size,
    "patch_num": df_patch_num,
    "gainloss": df_gainloss_all,
    "increase_decrease": df_inde_all,
    }
    #result = pattern, pattern_maps, map_title, generated_charts, chart_titles, outputs
    #result = pattern, outputs
    return outputs

def write_image(pattern,data,FileName):
    from dynamicpatch import WriteData
    WriteData.writedata(FileName, pattern,data,'byte')
    

