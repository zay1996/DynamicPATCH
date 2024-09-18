# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 14:27:07 2024

@author: AiZhang
"""

#%% OPTION 2: run code yourself
from dynamicpatch import main
## specify the parameters 
main.run_dynamicpatch(
        workpath = "D:/OneDrive - Clark University/Desktop/Research/patchmanuscript/inputs/lcm",
        year = [
        1971,
        1985,
        1999
    ],
        in_nodata = 0,
        connectivity = 8,
        targ_pre = 1,
        study_area = None,
        map_show = True,
        chart_show = True,
        #unit = 'sqm2', # let program decide automatically
        log_scale = True, 
        export_map = False,
        width = 0.35
    )


#%%
from dynamicpatch import main
## specify the parameters 
main.run_dynamicpatch(
        workpath = "D:/OneDrive - Clark University/Desktop/Research/patchmanuscript/inputs/examplev4.xlsx",
        year = [0,
                1
    ],
        in_nodata = -1,
        connectivity = 8,
        targ_pre = 1,
        study_area = None,
        map_show = True,
        chart_show = True,
        #unit = 'sqm2', # let program decide automatically
        log_scale = False, 
        export_map = False,
        width = 0.35
    )


#%%
from dynamicpatch import main
## specify the parameters 
main.run_dynamicpatch(
        workpath = "C:/Users/aiiyi/OneDrive - Clark University/Desktop/Research/patchmanuscript/inputs/lcm",
        year = [
        1971,
        1985,
        1999
    ],
        in_nodata = 0,
        connectivity = 8,
        targ_pre = 1,
        study_area = None,
        map_show = True,
        chart_show = True,
        #unit = 'sqm2', # let program decide automatically
        log_scale = True, 
        export_map = False,
        width = 0.35
    )
