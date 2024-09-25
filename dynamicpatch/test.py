# -*- coding: utf-8 -*-
"""
This script reads parameters and input data and run the analysis by calling other functions

@author: Aiyin Zhang
"""

from osgeo import gdal
import numpy as np
import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from dynamicpatch import WriteData
import glob

#%% OPTION 1: Use Interface 
import tkinter as tk
from dynamicpatch.config import read_params_interface
from dynamicpatch.interface import MapApp

proc_params, data, data_val = read_params_interface()

# Create the main Tkinter window
root = tk.Tk()
root.withdraw()  # Hide the root window

# Create and show the MapApp
map_app = MapApp(root)
map_app.protocol("WM_DELETE_WINDOW", root.quit)
root.mainloop()

#absence, presence, nodata, nt, nl, ns, connectivity = proc_params
      


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
        in_nodata = 0, # optional, default = 0
        connectivity = 8, # optional, default = 8
        targ_pre = 1, # optional, default = 1
        study_area = None, # optional, default = None
        map_show = True, # optional, default = True
        chart_show = True, # optional, default = True 
        unit = None, # let program decide automatically
        log_scale = True, # optional
        export_map = True, # optional, default = True
        width = 0.35 # optional, default = 0.35
    )