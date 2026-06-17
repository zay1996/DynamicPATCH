# -*- coding: utf-8 -*-
"""
the main script that runs the entire analysis.
reads parameters and input data and run the analysis by calling other functions

@author: Aiyin Zhang
"""

import importlib 
from dynamicpatch import config 
from dynamicpatch import config_new

def run_dynamicpatch(
        workpath,
        year,
        in_nodata = 0,
        connectivity = 8,
        targ_pre = 1,
        mask = None,
        study_area = None,
        map_show = True,
        chart_show = True,
        gif_show = False,
        unit = None, # let program decide automatically
        log_scale = True, 
        export_map = False,
        width = None,
        rotation = 45,
        res = None):
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
    export_map: str
        The directory of storing the transition maps. Default is None
    unit: String.
        Specifying the area unit for the result graphics. There are three options:
        'pixels','sqm2', and 'km2'. Default is 'Default' and the program will decide the 
        optimal area unit based on the size of the input data.
    log_scale: Boolean
        Whether the size distribution graph will be displayed in log scale 
        (the current version only support the size distribution graph to be displayed
         in logscale)
        Default is True.
    width: float
        The width of the bars in the two bar charts. Default is 0.35. 
    rotation: int
        The rotation angle of year labels in the stacked bar chart. Default is 0.
    res: int
        Spatial resolution of the input maps in meters. If not specified, the resolution is read automatically
    '''
    
    
    params, data, data_val = config_new.read_params\
        (workpath, year,targ_pre, connectivity,in_nodata, study_area,res)
    print("data loaded")
    from dynamicpatch import processing        
    importlib.reload(processing)  
    processing.initialize(params,data_val)
    result = processing.run_analysis(
                                     params,
                                     data,
                                     data_val,
                                     mask = mask,
                                     mapshow = map_show, 
                                     chartsshow = chart_show, 
                                     unit = unit, 
                                     export_map = export_map, 
                                     gif_map = gif_show,
                                     width = width, 
                                     log_scale = log_scale,
                                     rotation = rotation
                                     )

    
    return result 
        