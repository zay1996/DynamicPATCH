# -*- coding: utf-8 -*-
"""
Example code of running DynamicPATCH using interface
@author: AiZhang
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
import dynamicpatch.config_new as config_new
import importlib
importlib.reload(config_new)
import tkinter as tk
from dynamicpatch.config_new import read_params_interface
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
      
# %%
