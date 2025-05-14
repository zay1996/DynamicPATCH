

# %%
import importlib 
import dynamicpatch 
from dynamicpatch import config 
from dynamicpatch import config_new

importlib.reload(config_new)

#%% test 
from dynamicpatch import config_new, read_data
importlib.reload(config_new)
importlib.reload(read_data)
path = 'C:\\OneDrive - Clark University\\'
datapath = path + 'Desktop\\Research\\PIE\\RefMapComp\\'
file_name = 'PIE_classall_RF_3.tif'
workpath = datapath + file_name
targ_pre = 2
in_nodata = 0
year = [2010, 2012, 2014, 2016, 2018, 2021]
type_ = 'raster'
areaunit = 'km2'
weight = False # only supported for tabular data 
chunk_size = 1000
connectivity = 8
study_area = None

#%%

def run_dynamicpatch(
        workpath,
        year,
        in_nodata = 0,
        connectivity = 8,
        targ_pre = 1,
        study_area = None,
        map_show = True,
        chart_show = True,
        unit = None, # let program decide automatically
        log_scale = True, 
        export_map = False,
        width = 0.35,
        rotation = 45):
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
        'pixels','sqm2', and 'km2'. Default is 'Default' and the program will decide the 
        optimal area unit based on the size of the input data.
    log_scale: Boolean
        Whether the size distribution graph will be displayed in log scale 
        (the current version only support the size distribution graph to be displayed
         in logscale)
        Default is True.
    width: float
        The width of the bars in the two bar charts. Default is 0.35. 
    '''
    
    
    params, data_val = config_new.read_params_dask\
        (workpath, year,targ_pre, connectivity,in_nodata, study_area)
    from dynamicpatch import processing_dask        
    importlib.reload(processing_dask)  
    
    processing_dask.initialize(params,data_val)
    result = processing_dask.run_analysis(
                                     params,
                                     data_val,
                                     mapshow = map_show, 
                                     chartsshow = chart_show, 
                                     unit = unit, 
                                     export_map = export_map, 
                                     width = width, 
                                     log_scale = log_scale,
                                     rotation = rotation
                                     )

    
    return result,params 
# %%
