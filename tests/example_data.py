# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 14:09:22 2024

@author: AiZhang
"""





import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np




from dynamicpatch import main
## specify the parameters 
result_exp = main.run_dynamicpatch(
        workpath = "D:/OneDrive - Clark University/Desktop/Research/patchmanuscript/inputs/examplev4.xlsx",
        year = [0,
                1
    ],
        in_nodata = -1,
        connectivity = 8,
        targ_pre = 1,
        study_area = None,
        map_show = False,
        chart_show = False,
        #unit = 'sqm2', # let program decide automatically
        log_scale = False, 
        export_map = False,
        width = 0.35
    )

pattern_exp, _, _, _, _, outputs_exp = result_exp 
df_inde_all_exp,data_exp,dataval_exp,binary = outputs_exp

#%%

from dynamicpatch.config import in_params, proc_params
workpath, year, connectivity, targ_pre, in_nodata, FileType, dataset,study_area = in_params['workpath'],\
    in_params['years'],in_params['connectivity'],in_params['presence'], in_params['nodata'],\
        in_params['FileType'], in_params['dataset'],in_params['study_area']
        
absence, presence, nodata, nt, nl, ns, connectivity = proc_params 
#%%
#### MAP OF PRESENCE AND ABSENCE ####

def preabs_exp(binary,status,ax = None):
    
    binarylist = [''+status+' absence', '' + status + ' presence']
    colorlist = ['White','Black']
    cmap = colors.ListedColormap(colorlist)
    boundaries = [absence-.5,absence+.5,presence+.5]
    norm = colors.BoundaryNorm(boundaries, ncolors=len(colorlist), clip=True)
    
    map_ = binary.astype('str')
    map_[binary == absence] = 'a'
    map_[binary == presence] = 'p'
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(15,15))
    
    
    im = ax.imshow(binary, interpolation='none',cmap=cmap,norm=norm)
    
    
    # Add white gridlines
    # Add gridlines manually
    for edge in np.arange(-0.5, map_.shape[1], 1):
        ax.axvline(x=edge, color='silver', linewidth=2)
    
    for edge in np.arange(-0.5, map_.shape[0], 1):
        ax.axhline(y=edge, color='silver', linewidth=2)
    # Remove the major ticks
    ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    
    # Add pixel value label
    '''
    for i in range(map_.shape[0]):
        for j in range(map_.shape[1]):
            ax.text(j, i, map_[i, j], ha='center', va='center', color='grey',fontsize = 14)
    '''
    # Remove axis ticks and labels
    #ax.axis('off')
    
    patches = [mpatches.Patch(facecolor=colorlist[i], label=binarylist[i],edgecolor='black', linewidth=.5) for i in np.arange(len(binarylist))]

    ax.legend(handles=patches,loc='lower center', bbox_to_anchor=(0.5, -0.12), ncol = 2, borderaxespad=0. )
    if ax is None:
        return fig
    if ax is not None:
        return im

#%%
#### MAP OF OVERLAY ####

datacrosst = np.zeros((nl,ns),dtype = int)
         
datacrosst[(binary[0] == 1) & (binary[1] == 1)] = 0
datacrosst[(binary[0] == 2) & (binary[1] == 2)] = 1
datacrosst[(binary[0] == 1) & (binary[1] == 2)] = 2
datacrosst[(binary[0] == 2) & (binary[1] == 1)] = 3


def overlaymap(map_):
    binarylist = ['Stable Absence','Stable Presence','Gain','Loss']
    colorlist = ['#bdbdbd','#525252','Blue','sienna']
    cmap = colors.ListedColormap(colorlist)
    boundaries = [-.5,.5,1.5,2.5,3.5]
    norm = colors.BoundaryNorm(boundaries, ncolors=len(colorlist), clip=True)
    
    fig, ax = plt.subplots(figsize=(15,15))
    
    
    im = ax.imshow(map_, interpolation='none',cmap=cmap,norm=norm)
    
    
    # Add white gridlines
    # Add gridlines manually
    for edge in np.arange(-0.5, map_.shape[1], 1):
        ax.axvline(x=edge, color='white', linewidth=2)
    
    for edge in np.arange(-0.5, map_.shape[0], 1):
        ax.axhline(y=edge, color='white', linewidth=2)
    # Remove the major ticks
    ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    '''
    # Add pixel value labels
    for i in range(map_.shape[0]):
        for j in range(map_.shape[1]):
            ax.text(j, i, map_[i, j], ha='center', va='center', color='grey',fontsize = 14)
    '''
    # Remove axis ticks and labels
    #ax.axis('off')
    
    patches = [mpatches.Patch(color=colorlist[i], label=binarylist[i]) for i in np.arange(len(binarylist))]
    # put those patched as legend-handles into the legend
    ax.legend(handles=patches,loc='lower center', bbox_to_anchor=(0.5, -0.10), ncol = 4, borderaxespad=0. )

overlaymap(datacrosst)
#%%
####### MAP OF TRANSITION PATTERNS #######
from dynamicpatch.config import df_cat

def transit_exp(pattern,ax = None):
    
    map_ =  pattern[0].astype('int')
    
        
    categorylist = [f"{value} {type_}" for value, type_ in zip(df_cat.sort_values(by = 'Value')['Value'], \
                                                                df_cat.sort_values(by = 'Value')['Type'])][1:]
    
    
    colorlist = []
    for cat in categorylist:
        colors_ = df_cat.loc[df_cat['Type'] == cat[2:], 'Color']
        for color in colors_:
            colorlist.append(color)
    
    cmap = colors.ListedColormap(colorlist)
    boundaries = np.append(np.array(df_cat.sort_values(by = 'Value')['Value']) - 0.5, df_cat['Value'].max()+0.5)[1:]
    #boundaries = [-1.5,-0.5,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5]
    norm = colors.BoundaryNorm(boundaries, ncolors=11, clip=True)
    
    
    #### Dominant Land Cover ######
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(15,15))
    
    
    im = ax.imshow(map_, interpolation='nearest',cmap=cmap,norm=norm)
    
    # Add gridlines manually
    for edge in np.arange(-0.5, map_.shape[1], 1):
        ax.axvline(x=edge, color='white', linewidth=2)
    
    for edge in np.arange(-0.5, map_.shape[0], 1):
        ax.axhline(y=edge, color='white', linewidth=2)
    
    # Remove the major ticks
    ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    
    # Add pixel value labels
    for i in range(map_.shape[0]):
        for j in range(map_.shape[1]):
            ax.text(j, i, map_[i, j], ha='center', va='center', color='black')
    # Remove axis ticks and labels
    #ax.axis('off')
    
    patches = [mpatches.Patch(color=colorlist[i], label=categorylist[i]) for i in np.arange(len(categorylist))]
    
    if ax is None:
    # put those patched as legend-handles into the legend
        ax.legend(handles=patches,bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0. )
    
    if ax is None:
        return fig
    else: 
        return im 


#%%

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#import plotly.io as pio
#import plotly.tools as tls
from dynamicpatch.config import df_cat

filepath = 'D:\\OneDrive - Clark University\\Desktop\\Research\\patchmanuscript\\graphs\\'
# Create the figure and axes with increased gap between rows
fig = plt.figure(figsize=(15, 11))
ncol_ = 2
nrow_ = 2
gs = gridspec.GridSpec(ncol_, nrow_, height_ratios=[0.75, 1.5], width_ratios=[1.5, 1.5], hspace=0.2, wspace=0.1)

ax1 = plt.subplot(gs[0, 0])
ax2 = plt.subplot(gs[0, 1])
ax3 = plt.subplot(gs[1, :])
# Create the maps
preabs_exp(binary[0],'Initial',ax = ax1)
preabs_exp(binary[1], 'Final',ax = ax2)
transit_exp(pattern_exp, ax = ax3)

ax1.set_title('Initial Time', fontsize = 25)
ax2.set_title('Final Time', fontsize = 25)
ax3.set_title('Transition Types during a Time Interval',fontsize = 25)
# Create a single legend at the bottom of the figure with larger font size
categorylist = [f"{value} {type_}" for value, type_ in zip(df_cat.sort_values(by = 'Value')['Value'], \
                                                        df_cat.sort_values(by = 'Value')['Type'])][1:]
    
colorlist = [df_cat.loc[df_cat['Type'] == cat[2:], 'Color'].values[0] for cat in categorylist]
patches = [mpatches.Patch(color=colorlist[i], label=categorylist[i]) for i in range(len(categorylist))]

# Split the patches into three groups
group1 = patches[0:1] + [patches[-1]]
group2 = patches[1:5]
group3 = patches[5:9]

# Add legends for each group
fig.legend(handles=group1, loc='lower center', bbox_to_anchor=(0.5, 0.05), ncol=len(group1), fontsize=14,frameon = False)
fig.legend(handles=group2, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=len(group2), fontsize=14, frameon = False)
fig.legend(handles=group3, loc='lower center', bbox_to_anchor=(0.5, -.03), ncol=len(group3), fontsize=14, frameon = False)


#fig.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, 0.0001), ncol = 6,fontsize=14)
#plt.subplots_adjust(left=0.2, right=1, top=0.95, bottom=0.1)
plt.savefig(filepath + 'example_plot_new.png', bbox_inches='tight',format='png',dpi = 1200)
#plt.tight_layout()
plt.show()


#%%
import Statsnew
import stacked_bar

filepath = 'D:\\OneDrive - Clark University\\Desktop\\Research\\patchmanuscript\\graphs\\'
df_patch_num,df_patch_size, patch_ave, patch_median, patch_ave_q1,patch_ave_q3\
 = stacked_bar.countnumsize(pattern,areaunit = 'pixels')

# Create a new figure with specified size and gridspec for layout control
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

# Plot second figure (fig2) on the right
fig1 = stacked_bar.gainloss_stackedbars(pattern,df_patch_size,option = 'area',areaunit = 'pixels',ax = ax1)
ax1.set_title('(a)', loc='left', fontsize=20, weight='bold')  # Label (b) on the top left of ax2
# Plot first figure (fig1) on the left
fig2 = stacked_bar.inde_stackedbars(pattern,df_inde_all,ax = ax2)
ax2.set_title('(b)', loc='left', fontsize=20, weight='bold')  # Label (a) on the top left of ax1

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig(filepath + 'bar1_bar2.png', bbox_inches='tight',format='png',dpi=1200)  # Save the combined plot as PNG
plt.show()