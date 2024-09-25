# -*- coding: utf-8 -*-
"""
Example code of running DynamicPATCH using example data. 

@author: Aiyin Zhang
"""

#%% 
import os
import dynamicpatch 
from dynamicpatch import main
package_dir = os.path.dirname(os.path.abspath(dynamicpatch.__file__)) # find directory of the package    
## specify the parameters 
main.run_dynamicpatch(
        workpath = package_dir + '/static/example.xlsx',
        year = [0,
                1
    ],
        in_nodata = -1, # optional, default = 0
        connectivity = 8, # optional, default = 8
        targ_pre = 1, # optional, default = 1
        study_area = None, # optional, default = None
        map_show = True, # optional, default = True
        chart_show = True, # optional, default = True 
        unit = None, # let program decide automatically
        log_scale = True, # optional
        export_map = False, # optional, default = True
        width = 0.35 # optional, default = 0.35
    )


