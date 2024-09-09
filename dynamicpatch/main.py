# -*- coding: utf-8 -*-
"""
the main script that runs the entire analysis.
reads parameters and input data and run the analysis by calling other functions

@author: Aiyin Zhang
"""


from dynamicpatch import config 

def run_dynamicpatch(
        workpath = None,
        year = None,
        in_nodata = 0,
        connectivity = 8,
        targ_pre = 1,
        study_area = None,
        map_show = True,
        chart_show = True,
        unit = None, # let program decide automatically
        log_scale = True, 
        export_map = False,
        width = 0.35):
    '''
    Run dynamic patch analysis in commandline 
    

    Parameters
    ----------
    workpath : string
        The path to the input data, could be a folder or the file itself. 
        Example: 'C:/User/Analysis/piemarsh.tif' 
    year: list
        List of years of the input data separated by comma. 
        Example: [1938,1971,2013]
    in_nodata: int
        No data value of the input data. Default if 0. 
    connectivity: int
        Case of connectivity. There are 2 options: 4 (4-connectivity case, or 
        the Rook's case) or 8 (8-connectivity case, or the queen's case). 
        Default is 8.
    targ_pre: int
        Value of the presence category. Default is 1. 
    Study area: string
        Name of the study area. Default is None.
    map_show: Boolean
        Whether the result map will be generated. Default is True. 
    chart_show: Boolean
        Whether the result graphics will be generated. Default is True.
    export_map: Boolean
        Whether the result map will be exported as a tif file. Default is False.
    unit: String.
        Specifying the area unit for the result graphics. There are three options:
        'pixels','sqm2', and 'km2'. Default is None and the program will decide the 
        optimal area unit based on the size of the input data.
    log_scale: Boolean
        Whether the size distribution graph will be displayed in log scale 
        (the current version only support the size distribution graph to be displayed
         in logscale)
        Default is True.
    width: float
        The width of the bars in the two bar charts. Default is 0.35. 
    '''
    
    
    proc_params, data, data_val = config.read_params\
        (workpath, year,targ_pre, connectivity,in_nodata, study_area)
        
    from dynamicpatch import processing     
    processing.initialize()
    processing.run_analysis(
        mapshow = map_show, chartsshow = chart_show, unit = unit, export_map = export_map, width = width, log_scale = log_scale)



        