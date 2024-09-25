# -*- coding: utf-8 -*-
"""
Create Maps associated with DynamicPATCH

@author: Aiyin Zhang
"""

import os
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import numpy as np
import dynamicpatch 
#from dynamicpatch import processing        
#import importlib 
#importlib.reload(processing)  
from dynamicpatch.config import year, df_cat
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox



def map_timepoint(tp,binary,absence,presence,res = None):
    '''
    Create initial maps of absence and presence at each time point

    Parameters
    ----------
    tp : int
        index for the time point to be mapped. e.g. tp for the first time point
        should be 0
    binary : 2d array
        Binary map of the target time point, including no data values
    absence : int
        Value of absence
    presence : int
        Value of presence=

    Returns
    -------
    fig : Figure
        Output map

    '''
    #### MAP OF PRESENCE AND ABSENCE ####
    binarylist = ['No Data','Absence','Presence']
    colorlist = ['White','Silver','Black']
    cmap = colors.ListedColormap(colorlist)
    boundaries = [absence-1.5,absence-.5,absence+.5,presence+.5]
    norm = colors.BoundaryNorm(boundaries, ncolors=len(colorlist), clip=True)

    map_ = binary.astype('int')

    fig, ax = plt.subplots(figsize=(15,15))
    ax.axis("off")


    im = ax.imshow(map_, interpolation='none',cmap=cmap,norm=norm)
    patches = []
    for i in range(len(binarylist)):
        if i == 0:  # Assume first entry is "No Data"
            patch = mpatches.Patch(facecolor=colorlist[i], label=binarylist[i], edgecolor='black', linewidth=.5)
        else:
            patch = mpatches.Patch(color=colorlist[i], label=binarylist[i])
        patches.append(patch)
        
    # put those patched as legend-handles into the legend
    ax.legend(handles=patches,bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0. )
    # Add north arrow SVG
    if res is not None and res != 0:
        package_dir = os.path.dirname(os.path.abspath(dynamicpatch.__file__)) # find directory of the package    
        north_arrow = package_dir+'/static/northarrow2.png'  # Update with the path to your SVG file
        img = Image.open(north_arrow)
        imagebox = OffsetImage(img, zoom=0.2)  # Adjust zoom as needed
        ab = AnnotationBbox(imagebox, (1.15, 0.1), frameon=False, xycoords='axes fraction', boxcoords="axes fraction", pad=0.0)
        ax.add_artist(ab)
    ax.set_title(str(year[tp]),fontsize = 20)   
    #fig.tight_layout()
    
    return fig 


def pattern_map(tp,pattern,data=None, res=None, ax = None,frame = 'off', north_arrow = True, type_='change'):
    '''
    Create transition pattern maps 

    Parameters
    ----------
    tp : int
        index for the time point to be mapped. e.g. tp for the first time point
        should be 0
    pattern : 3d array
        Transition pattern map for all time intervals.
    data : TYPE, optional
        DESCRIPTION. The default is None.
    res : int, optional
        Resolution of the input in meters. The default is None.
    ax : axis, optional
        Figure axis, use when plot as a subfigure. The default is N
    frame: string, optional
        Whether a frame around the map is shown. The default is 'off'
    north_arrow: boolean, optional
        Whether a north arrow is plotted. The default is True 
    type_ : String, optional
        Specify whether the map if for change analysis or comparison analysis. 
        The default is 'change'.

    Returns
    -------
    fig: Figure
        Stacked bar chart figure.

    '''

    categorylist = list(df_cat.sort_values(by='Value')['Type'])
    colorlist = []
    for cat in categorylist:
        colors_ = df_cat.loc[df_cat['Type'] == cat, 'Color']
        for color in colors_:
            colorlist.append(color)
    
    cmap = colors.ListedColormap(colorlist)
    boundaries = np.append(np.array(df_cat.sort_values(by='Value')['Value']) - 0.5, df_cat['Value'].max() + 0.5)
    norm = colors.BoundaryNorm(boundaries, ncolors=11, clip=True)
    # Get geographic information from the GDAL dataset
    if data is not None:
        geo_transform = data.GetGeoTransform()
        x_origin = geo_transform[0]
        y_origin = geo_transform[3]
        pixel_width = geo_transform[1]
        pixel_height = geo_transform[5]
    
        # Calculate extent in the original projection units
        x_max = x_origin + pixel_width * pattern[tp].shape[1]
        y_min = y_origin + pixel_height * pattern[tp].shape[0]
        extent = [x_origin,x_max,y_min,y_origin]
    if ax is not None:
        flag_ax = True
    
    if ax is None:
        flag_ax = False
        fig, ax = plt.subplots(figsize=(15, 15))
        #ax.axis('off')
    if (frame == 'off'):
        ax.axis('off')

    
    if data is not None:
        im = ax.imshow(pattern[tp], interpolation='none', cmap=cmap, norm=norm,extent = extent)
    
    if data is None:
        im = ax.imshow(pattern[tp], interpolation='none', cmap=cmap, norm=norm)


    # Set coordinate labels
    if data is not None:
        ax.set_xticks(np.linspace(x_origin, x_max, 10))  
        ax.set_yticks(np.linspace(y_min, y_origin, 10))  

    
    patches = []
    for i in range(len(categorylist)):
        if i == 0:  # Assume first entry is "No Data"
            patch = mpatches.Patch(facecolor=colorlist[i], label=categorylist[i], edgecolor='black', linewidth=.5)
        else:
            patch = mpatches.Patch(color=colorlist[i], label=categorylist[i])
        patches.append(patch)
    
    if (flag_ax == False):
        ax.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
    
    if res is not None:
        scalebar = ScaleBar(res, location='lower right')  # 1 pixel = 2 meter
        ax.add_artist(scalebar)

    # Add north arrow SVG
    if north_arrow is True:
        if res is not None and res != 0:
            package_dir = os.path.dirname(os.path.abspath(dynamicpatch.__file__)) # find directory of the package    
            north_arrow = package_dir+'/static/northarrow2.png'  # Update with the path to your SVG file
            img = Image.open(north_arrow)
            imagebox = OffsetImage(img, zoom=0.3)  # Adjust zoom as needed
            ab = AnnotationBbox(imagebox, (1.10, 0.1), frameon=False, xycoords='axes fraction', boxcoords="axes fraction", pad=0.0)
            ax.add_artist(ab)
    #fig.tight_layout()
    
    if (flag_ax == False):
        if(type_ == 'compare'):
            from dynamicpatch.main import compare
            ax.set_title(compare[0]+'vs.'+compare[1]+str(year[tp]))    
        if(type_ == 'change'):
            ax.set_title(str(year[tp]) + ' - ' + str(year[tp+1]),fontsize = 20)    
            
            
    if (flag_ax == False):
        return fig
    else:
        return im
    

    
