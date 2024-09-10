# -*- coding: utf-8 -*-
"""
This script calls the interface, run the entire script and use the interface to
complete the analysis. 

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


      


