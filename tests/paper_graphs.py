# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 16:02:02 2024

@author: AiZhang
"""

#%%
from osgeo import gdal
import numpy as np
import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import glob

#%% pond 
from dynamicpatch import main
## specify the parameters 
result_pond = main.run_dynamicpatch(
        workpath = os.path.dirname(os.path.dirname(os.getcwd())) + '/inputs/pondbinary.tif',
        year = [
        1938,
        1971,
        2013
    ],
        in_nodata = -1,
        connectivity = 8,
        targ_pre = 1,
        study_area = None,
        map_show = True,
        chart_show = True,
        unit = 'sqm2', # let program decide automatically
        log_scale = False, 
        export_map = False,
        width = 0.35
    )

pondpattern, _, _, _, _, pond_outputs = result_pond 
df_inde_all_pond,data_pond,dataval_pond, binary_pond = pond_outputs

#%% marsh
## specify the parameters 
from dynamicpatch import main
result_marsh = main.run_dynamicpatch(
        workpath = "D:/OneDrive - Clark University/Desktop/Research/patchmanuscript/inputs/marshbinary.tif",
        year = [
        1938,
        1971,
        2013
    ],
        in_nodata = -1,
        connectivity = 8,
        targ_pre = 1,
        study_area = None,
        map_show = True,
        chart_show = True,
        unit = 'sqm2', # let program decide automatically
        log_scale = True, 
        export_map = False,
        width = 0.35
    )

marshpattern, _, _, _, _, marsh_outputs = result_marsh 
df_inde_all_marsh,data_marsh,dataval_marsh, binary_marsh = marsh_outputs
    
#%% Create study area map 
filepath = 'D:\\OneDrive - Clark University\\Desktop\\Research\\patchmanuscript\\graphs\\'

import matplotlib.colors as colors
import matplotlib.patches as mpatches
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.patches import FancyArrow
from dynamicpatch.config import res 

# Create the figure and axes with increased gap between rows
fig = plt.figure(figsize=(20, 10))
ncol_ = 3
nrow_ = 1
gs = fig.add_gridspec(nrows = nrow_, ncols = ncol_, height_ratios=[1.5], width_ratios=[1, 1, 1], hspace=0.1, wspace=0.1)

from dynamicpatch import read_data
filedir = 'D:\\CLASS\\GEOG 379\\ponds\\HR\\'
FilePathl = [filedir+'LC1938.tif',filedir+'LC1972.tif',filedir+'LC2013.tif']
data, dataar,size = read_data.readdatafunc('Tif', FilePathl)
nl,ns = size
pie = np.zeros((3,nl,ns)).astype('byte')
pie[0:2][dataar[0:2] == 1] =1 # 1 = pond
pie[0:2][dataar[0:2] == 3] = 3 # 3 = river
pie[0:2][dataar[0:2] == 2] = 4 # 4 = upland
pie[0:2][dataar[0:2] == 4] = 2 # 2 = marsh 

pie[2][dataar[2]==2]=1
pie[2][dataar[2]==4]=3
pie[2][dataar[2]==3]=4
pie[2][dataar[2]==1]=2


pie[dataar == 0] = 0 # -1 = nodata 

year = [1938,1971,2013]
lclist = ['No Data','Pond','Marsh','River','Upland']
colorlist = ['White','aqua','Yellow','#3182bd','#31a354']
cmap = colors.ListedColormap(colorlist)
boundaries = np.arange(0,6,1) - .5
norm = colors.BoundaryNorm(boundaries, ncolors=len(colorlist), clip=True)

for i in range(3):
    ax = plt.subplot(gs[0, i])
    ax.imshow(pie[i],interpolation = 'none',cmap = cmap, norm = norm)
    ax.axis("off")
    ax.set_title(str(year[i]), fontsize = 30)
    if(i==2):
        patches = [mpatches.Patch(color=colorlist[i], label=lclist[i]) for i in np.arange(len(lclist))]
        # put those patched as legend-handles into the legend
        ax.legend(handles=patches,bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0. )
        scalebar = ScaleBar(res, location='lower right')  # 1 pixel = 2 meter
        ax.add_artist(scalebar)


#plt.savefig(filepath + 'studyarea'+'map.png',  bbox_inches='tight',format='png', dpi = 600) 
#%% generate result graph, without zoom in 
from dynamicpatch import create_charts
from dynamicpatch import create_maps
import matplotlib.patches as mpatches
from dynamicpatch.config import df_cat, year, res
# Create the figure and axes with increased gap between rows
fig = plt.figure(figsize=(18, 20))
ncol_ = 2
nrow_ = 2
gs = fig.add_gridspec(ncol_+1, nrow_+1, height_ratios=[0.001, 1.5, 1.5], width_ratios=[1.5, 1.5, 0.001], hspace=0.1, wspace=0.1)

ax1 = plt.subplot(gs[1, 0])
ax2 = plt.subplot(gs[1, 1])
ax3 = plt.subplot(gs[2, 0])
ax4 = plt.subplot(gs[2, 1])
# Create the maps
create_maps.pattern_map(0,marshpattern, res = res, ax=ax1)
create_maps.pattern_map(1,marshpattern, res = res, ax=ax2)
create_maps.pattern_map(0,pondpattern, res = res, ax=ax3)
create_maps.pattern_map(1,pondpattern, res = res, ax=ax4)

ax1.set_title(str(year[0]) + ' - ' + str(year[1]), fontsize = 30)
ax2.set_title(str(year[1]) + ' - ' + str(year[2]), fontsize = 30)

ax1.text(-0.05, 0.5, 'Marsh', fontsize=30, va='center', ha='right', transform=ax1.transAxes)
ax3.text(-0.05, 0.5, 'Pond', fontsize=30, va='center', ha='right', transform=ax3.transAxes)
# Create a single legend at the bottom of the figure with larger font size
categorylist = list(df_cat.sort_values(by='Value')['Type'])
colorlist = [df_cat.loc[df_cat['Type'] == cat, 'Color'].values[0] for cat in categorylist]
patches = [mpatches.Patch(color=colorlist[i], label=categorylist[i]) for i in range(len(categorylist))]
fig.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, 0.001), ncol = 6,fontsize=16)
plt.subplots_adjust(bottom=0.05)
#plt.tight_layout()
plt.show()


#%% generate result graph, with zoom in 
filepath = 'D:\\OneDrive - Clark University\\Desktop\\Research\\patchmanuscript\\graphs\\'
from dynamicpatch import create_charts
from dynamicpatch import create_maps
import matplotlib.patches as mpatches
import matplotlib.colors as colors
from dynamicpatch.config import df_cat
from dynamicpatch.config import in_params, proc_params, data_val, res
workpath, year, connectivity, targ_pre, in_nodata, FileType, dataset,study_area = in_params['workpath'],\
    in_params['years'],in_params['connectivity'],in_params['presence'], in_params['nodata'],\
        in_params['FileType'], in_params['dataset'],in_params['study_area']

params = proc_params
absence, presence, nodata, nt, nl, ns, connectivity = params
# Create the figure and axes with increased gap between rows
fig = plt.figure(figsize=(24, 34))
ncol_ = 6
nrow_ = 4
gs = fig.add_gridspec(nrow_, ncol_, height_ratios=[1.5, 0.5, 1.5, 0.5], \
                      width_ratios=[0.5,0.5,0.5,0.5,0.5,0.5], hspace=0.1, wspace=0.1)
categorylist = list(df_cat.sort_values(by='Value')['Type'])
for cat in categorylist:
    colors_ = df_cat.loc[df_cat['Type'] == cat, 'Color']
    for color in colors_:
        colorlist.append(color)

cmap = colors.ListedColormap(colorlist)
    
patternall = np.zeros((4,nl,ns),dtype = 'int')
patternall[0:2,:,:] = marshpattern
patternall[2:4,:,:] = pondpattern 

datavalall = np.zeros((6,nl,ns),dtype = 'ubyte')
datavalall[0:3,:,:] = dataval_marsh
datavalall[3:6,:,:] = dataval_pond 


ax_main = {}
ax_zoom = {}
ax_main[0] = plt.subplot(gs[0, 0:3])
ax_main[1] = plt.subplot(gs[2, 0:3])
ax_main[2] = plt.subplot(gs[0, 3:6])
ax_main[3] = plt.subplot(gs[2, 3:6])

ax_bi_0 = {}
ax_bi_0[0] = plt.subplot(gs[1,0])
ax_bi_0[1] = plt.subplot(gs[3,0])
ax_bi_0[2] = plt.subplot(gs[1,3])
ax_bi_0[3] = plt.subplot(gs[3,3])

ax_bi_1 = {}
ax_bi_1[0] = plt.subplot(gs[1,1])
ax_bi_1[1] = plt.subplot(gs[3,1])
ax_bi_1[2] = plt.subplot(gs[1,4])
ax_bi_1[3] = plt.subplot(gs[3,4])

ax_tr = {}
ax_tr[0] = plt.subplot(gs[1,2])
ax_tr[1] = plt.subplot(gs[3,2])
ax_tr[2] = plt.subplot(gs[1,5])
ax_tr[3] = plt.subplot(gs[3,5])

# Create the maps


create_maps.pattern_map(0,marshpattern, res = res, ax=ax_main[0],north_arrow = False)
create_maps.pattern_map(1,marshpattern, res = res, ax=ax_main[1])
create_maps.pattern_map(0,pondpattern, res = res, ax=ax_main[2], north_arrow = False)
create_maps.pattern_map(1,pondpattern, res = res, ax=ax_main[3])


for i in range(4):
    ax_main[i].axis('off')

ax_main[0].set_title('Marsh', fontsize = 30)
ax_main[2].set_title('Pond', fontsize = 30)

ax_main[0].text(-0.05, 0.5, str(year[0]) + ' - ' + str(year[1]), fontsize=30, va='center', ha='right', transform=ax_main[0].transAxes)
ax_main[1].text(-0.05, 0.5, str(year[1]) + ' - ' + str(year[2]), fontsize=30, va='center', ha='right', transform=ax_main[1].transAxes)

ax_main[0].set_title('(a)', loc='left', fontsize=20, weight='bold')  # Label (b) on the top left of ax2
ax_main[1].set_title('(c)', loc='left', fontsize=20, weight='bold')  # Label (b) on the top left of ax2
ax_main[2].set_title('(b)', loc='left', fontsize=20, weight='bold')  # Label (b) on the top left of ax2
ax_main[3].set_title('(d)', loc='left', fontsize=20, weight='bold')  # Label (b) on the top left of ax2


# Create a single legend at the bottom of the figure with larger font size
categorylist = list(df_cat.sort_values(by='Value')['Type'])
colorlist = [df_cat.loc[df_cat['Type'] == cat, 'Color'].values[0] for cat in categorylist]



zoom_areas = [
    (100, 1900),  # Example positions for zoom areas, adjust as needed
    (1300, 900),
    (100,1900),
    (1300,900)
    #(1800, 500)
]   

zoom_in_size_x = 200  # Size of the zoom-in area
zoom_in_size_y = 200    

      
zoom_colors = ['magenta', 'green', 'magenta','green'] 

#labels = ['(a)','(b)','(c)','(d)',]

for i, (x, y) in enumerate(zoom_areas):
    rect = mpatches.Rectangle((y, x), zoom_in_size_y, zoom_in_size_x, fill=False, edgecolor=zoom_colors[i], linewidth=5)
    ax_main[i].add_patch(rect)

# Plot presence and absence maps and zoom-in views
binary_colorlist = ['white', 'black']
binarylist = ['Absence','Presence']
binary_cmap = colors.ListedColormap(binary_colorlist)
binary_boundaries = [0.5, 1.5, 2.5]
binary_norm = colors.BoundaryNorm(binary_boundaries, binary_cmap.N)

for i, (x, y) in enumerate(zoom_areas):

    binary = np.zeros((2,nl,ns),dtype = 'ubyte')
    if(i<2):
        binary[dataval_marsh[i:i+2,:,:] == targ_pre] = presence
        binary[dataval_marsh[i:i+2,:,:] != targ_pre] = absence
        binary[dataval_marsh[i:i+2,:,:] == in_nodata] = nodata
    if(i>=2):
        binary[dataval_pond[i-2:i,:,:] == targ_pre] = presence
        binary[dataval_pond[i-2:i,:,:] != targ_pre] = absence
        binary[dataval_pond[i-2:i,:,:] == in_nodata] = nodata        
    # Presence and absence at first time point
    ax_binary1 = fig.add_subplot(ax_bi_0[i])
    binary_view1 = binary[0][x:x + zoom_in_size_x, y:y + zoom_in_size_y]
    ax_binary1.imshow(binary_view1, cmap=binary_cmap, norm=binary_norm)
    patches = []
    for b, color in enumerate(binary_colorlist):
        if color == 'white':
            patches.append(mpatches.Patch(facecolor='white', edgecolor='black', label=binarylist[b], linewidth=1.5))
        else:
            patches.append(mpatches.Patch(color=color, label=binarylist[b]))
    ax_binary1.legend(handles=patches,bbox_to_anchor=(0.00, 0), loc=2, ncol = 2,borderaxespad=0. )
    if(i < 2):
        ax_binary1.set_title(str(year[i]),fontsize = 20)
    if(i >=2):
        ax_binary1.set_title(str(year[i-2]),fontsize = 20)
    ax_binary1.set_xticks([])
    ax_binary1.set_yticks([])

    # Presence and absence at second time point
    ax_binary2 = fig.add_subplot(ax_bi_1[i])
    binary_view2 = binary[1][x:x + zoom_in_size_x, y:y + zoom_in_size_y]
    ax_binary2.imshow(binary_view2, cmap=binary_cmap, norm=binary_norm)
    if(i < 2):
        ax_binary2.set_title(str(year[i+1]),fontsize = 20)
    if(i >=2):
        ax_binary2.set_title(str(year[i-1]),fontsize = 20)
    ax_binary2.set_xticks([])
    ax_binary2.set_yticks([])

    # Plot zoom-in view
    ax_zoom = fig.add_subplot(ax_tr[i])  # Span the zoom-in views across two columns
    zoom_view = patternall[i][x:x + zoom_in_size_x, y:y + zoom_in_size_y]
    #ax_zoom.imshow(zoom_view, interpolation='none', cmap=cmap, norm=norm)
    create_maps.pattern_map(i, patternall[:,x:x + zoom_in_size_x, y:y + zoom_in_size_y], res = res, ax=ax_zoom, north_arrow = False)
    ax_zoom.set_title('Transition Pattern',fontsize = 20)
    ax_zoom.set_xticks([])
    ax_zoom.set_yticks([])
    # Set border color for zoom-in view
    for spine in ax_zoom.spines.values():
        spine.set_edgecolor(zoom_colors[i])
        spine.set_linewidth(5)
    
    #ax_binary1.text(-0.05, 1.05, labels[i], transform=ax_binary1.transAxes, fontsize=20, fontweight='bold', va='top', ha='right')
patches = [mpatches.Patch(color=colorlist[i], label=categorylist[i]) for i in range(len(categorylist))][1:]
fig.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, 0.001), ncol = 6,fontsize=16)
plt.subplots_adjust(bottom=0.05)
#plt.tight_layout()
plt.savefig(filepath + 'pondmarsh'+'mapNEW.tif', dpi = 600, bbox_inches='tight',format='tif') 
plt.show()



#%%
## stacked bar
filepath = 'D:\\OneDrive - Clark University\\Desktop\\Research\\patchmanuscript\\graphs\\'
import stacked_bar
df_patch_num_marsh,df_patch_size_marsh, patch_ave_marsh, patch_median_marsh, patch_ave_q1_marsh,patch_ave_q3_marsh = stacked_bar.countnumsize(marshpattern, type_ = type_,areaunit = 'km2')
df_patch_num_pond,df_patch_size_pond, patch_ave_pond, patch_median_pond,patch_ave_q1_pond,patch_ave_q3_pond = stacked_bar.countnumsize(pondpattern, type_ = type_,areaunit = 'km2')

# Create a new figure with specified size and gridspec for layout control
fig, axes = plt.subplots(2, 2, figsize=(26, 12))


fig1 = stacked_bar.gainloss_stackedbars(marshpattern,df_patch_size_marsh,option = 'area',legend = 'no',ax = axes[0,0],areaunit = 'km2')
axes[0,0].set_title('(a)', loc='left', fontsize=20, weight='bold')  # Label (b) on the top left of ax2

fig2 = stacked_bar.inde_stackedbars(marshpattern,df_inde_all_marsh,ax = axes[1,0],legend = 'no')
axes[1,0].set_title('(c)', loc='left', fontsize=20, weight='bold')  # Label (a) on the top left of ax1


fig3 = stacked_bar.gainloss_stackedbars(pondpattern,df_patch_size_pond,option = 'area',ax = axes[0,1], areaunit = 'km2')
axes[0,1].set_title('(b)', loc='left', fontsize=20, weight='bold')  # Label (b) on the top left of ax2

fig4 = stacked_bar.inde_stackedbars(pondpattern,df_inde_all_pond,ax = axes[1,1])
axes[1,1].set_title('(d)', loc='left', fontsize=20, weight='bold')  # Label (a) on the top left of ax1

axes[0,0].text(0.5, 1.1, 'Marsh', fontsize=30, va='top', ha='center', transform=axes[0,0].transAxes)
axes[0,1].text(0.5, 1.1, 'Pond', fontsize=30, va='top', ha='center', transform=axes[0,1].transAxes)


# Adjust layout and save the figure
plt.tight_layout()
#plt.savefig(filepath + 'pondmarsh'+'stackedbar.png', format='png',dpi = 1200)  # Save the combined plot as PNG
plt.show()

    
#%% get tables for area 
perc_size_loss_marsh_ave = (df_patch_size_marsh.iloc[:,6:-1].sum()/df_patch_size_marsh.iloc[:,6:-1].sum().sum())*100
perc_size_loss_marsh = df_patch_size_marsh.iloc[:,6:-1].div(df_patch_size_marsh.iloc[:,6:-1].sum(axis = 1),axis = 0)


perc_size_loss_marsh.to_csv(filepath+ 'marshperc.csv')
perc_size_gain_marsh = (df_patch_size_marsh.iloc[:,2:6].sum()/df_patch_size_marsh.iloc[:,2:6].sum().sum())*100

perc_size_loss_pond_ave = (df_patch_size_pond.iloc[:,6:-1].sum()/df_patch_size_pond.iloc[:,6:-1].sum().sum())*100
perc_size_loss_pond = df_patch_size_pond.iloc[:,6:-1].div(df_patch_size_pond.iloc[:,6:-1].sum(axis = 1),axis = 0)
perc_size_loss_pond.to_csv(filepath+ 'pondperc.csv')


#%% get tables for number of increase decrease
_,df_inde_all_marsh_annual,_,_ = stacked_bar.inde_table(marshpattern,df_inde_all_marsh) 
_,df_inde_all_pond_annual,_,_ = stacked_bar.inde_table(pondpattern, df_inde_all_pond)

df_inde_all_marsh_annual.to_csv(filepath+'marshindeann.csv')
df_inde_all_pond_annual.to_csv(filepath+'pondindeann.csv')


inde_ave = df_inde_all_pond.iloc[:,1:].abs().sum()/df_inde_all_pond.iloc[:,1:].abs().sum().sum()

#%% export area table to csv
df_patch_size_marsh.to_csv(filepath + 'marsharea.csv')
df_patch_size_pond.to_csv(filepath + 'pondarea.csv')

#%% bar charts 
import stacked_bar
df_patch_num_marsh,df_patch_size_marsh, patch_ave_marsh, patch_median_marsh, patch_ave_q1_marsh,patch_ave_q3_marsh = stacked_bar.countnumsize(marshpattern, type_ = type_)
df_patch_num_pond,df_patch_size_pond, patch_ave_pond, patch_median_pond,patch_ave_q1_pond,patch_ave_q3_pond = stacked_bar.countnumsize(pondpattern, type_ = type_)


fig, axes = plt.subplots(2, 2, figsize=(26, 14))

fig1 = stacked_bar.plot_num(df_patch_num_marsh,year, width = 0.35, type_ = type_,ax = axes[0,0])
axes[0,0].set_title('(a)', loc='left', fontsize=20, weight='bold')  # Label (b) on the top left of ax2
fig2 = stacked_bar.plot_ave_size(patch_ave_marsh, patch_median_marsh,patch_ave_q1_marsh,patch_ave_q3_marsh, \
                                 year,type_ = type_, log_scale = True, width = 0.35,ax = axes[1,0])
axes[0,1].set_title('(b)', loc='left', fontsize=20,weight='bold')  # Label (b) on the top left of ax2

fig3 = stacked_bar.plot_num(df_patch_num_pond,year, width = 0.35, type_ = type_,ax = axes[0,1])
axes[1,0].set_title('(c)', loc='left', fontsize=20, weight='bold') 
fig4 = stacked_bar.plot_ave_size(patch_ave_pond, patch_median_pond,patch_ave_q1_pond,patch_ave_q3_pond, \
                                 year,type_ = type_, log_scale = True, width = 0.35,ax = axes[1,1])
axes[1,1].set_title('(d)', loc='left', fontsize=20, weight='bold') 

axes[0,0].text(0.5, 1.1, 'Marsh', fontsize=30, va='top', ha='center', transform=axes[0,0].transAxes)
axes[0,1].text(0.5, 1.1, 'Pond', fontsize=30, va='top', ha='center', transform=axes[0,1].transAxes)

## to do: revise map, make bar chart (add mean), update stacked bar. make table, make a line chart showing net change 

plt.tight_layout()
plt.savefig(filepath + 'pondmarsh'+'statsbar.pdf', format='pdf')  # Save the combined plot as PNG
plt.show()


#%% export number table to csv
df_patch_num_marsh.to_csv(filepath + 'marshnum.csv')
df_patch_num_pond.to_csv(filepath + 'pondnum.csv')

#%% net change in number of patches

from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
binary_marsh = np.zeros((3,nl,ns),dtype = 'uint8')
binary_marsh[dataval_marsh == targ_pre] = 1
binary_marsh[dataval_marsh != targ_pre] = 0
binary_marsh[dataval_marsh == in_nodata] = 0

binary_pond = np.zeros((3,nl,ns),dtype = 'uint8')
binary_pond[dataval_pond == targ_pre] = 1
binary_pond[dataval_pond != targ_pre] = 0
binary_pond[dataval_pond == in_nodata] = 0

num_m, num_p = [],[]
size_m,size_p = [],[]


for i,y in enumerate(year):
    num_labels_m, patchlabels_m = cv2.connectedComponents(binary_marsh[i], connectivity=connectivity)
    num_labels_p, patchlabels_p = cv2.connectedComponents(binary_pond[i], connectivity=connectivity)
    num_m.append(num_labels_m)
    num_p.append(num_labels_p)
    size_m.append(np.sum(binary_marsh[i]))
    size_p.append(np.sum(binary_pond[i]))

size_m = [m*(res**2)/(1000000) for m in size_m]
size_p = [p*(res**2)/(1000000) for p in size_p]
# Create figure and subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

# X-axis positions
x = np.arange(len(year))

# Width of the bars
width = 0.35

# Plot for marsh
bars1 = ax1.bar(x, num_m, width, color='silver', edgecolor='black', label='Number of Patches')
ax1.set_xlabel('Year')
ax1.set_ylabel('Number of Patches')
ax1.set_xticks(x)
ax1.set_xticklabels(year)
ax1.set_title('Marsh')

# Create a second y-axis for area of presence
ax1_2 = ax1.twinx()
line1, = ax1_2.plot(x, size_m, color='black', marker='o', label='Area of Presence')
ax1_2.set_ylabel('Area of Presence (km²)')
# Get the current y-axis limits
_, ymax = ax1_2.get_ylim()
ax1_2.set_ylim(0, ymax+1)

# Plot for pond
bars2 = ax2.bar(x, num_p, width, color='silver', edgecolor='black', label='Number of Patches')
ax2.set_xlabel('Year')
ax2.set_ylabel('Number of Patches')
ax2.set_xticks(x)
ax2.set_xticklabels(year)
ax2.set_title('Pond')

# Create a second y-axis for area of presence
ax2_2 = ax2.twinx()
line2, = ax2_2.plot(x, size_p, color='black', marker='o', label='Area of Presence')
ax2_2.set_ylabel('Area of Presence (km²)')
_, ymax = ax2_2.get_ylim()
ax2_2.set_ylim(0, ymax)
# Combine legends
bars_legend = mpatches.Patch(facecolor = 'silver',edgecolor='black', label='Number of Patches')
lines_legend = Line2D([], [], color='black', marker='o', linestyle='-', label='Area of Presence')

legend = fig.legend(handles=[bars_legend, lines_legend], loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=12)
legend.get_frame().set_linewidth(0) 
plt.tight_layout()
filepath = 'D:\\OneDrive - Clark University\\Desktop\\Research\\patchmanuscript\\graphs\\'
plt.savefig(filepath + 'pondmarsh'+'netchange.tif', bbox_inches='tight',format='tif')


#%% net change in number of patches

from matplotlib.lines import Line2D
import matplotlib.patches as mpatches


binary_pond = np.zeros((3,nl,ns),dtype = 'uint8')
binary_pond[dataval_pond == targ_pre] = 1
binary_pond[dataval_pond != targ_pre] = 0
binary_pond[dataval_pond == in_nodata] = 0

num_m, num_p = [],[]
size_m,size_p = [],[]


for i,y in enumerate(year):
    num_labels_p, patchlabels_p = cv2.connectedComponents(binary_pond[i], connectivity=connectivity)
    num_p.append(num_labels_p)
    size_p.append(np.sum(binary_pond[i]))


size_p = [p*(res**2)/(1000000) for p in size_p]
# Create figure and subplots
fig, ax = plt.subplots(1, 1, figsize=(7, 4))

# X-axis positions
x = np.arange(len(year))

# Width of the bars
width = 0.35



# Plot for pond
bars2 = ax.bar(x, num_p, width, color='silver', edgecolor='black', label='Number of Patches')
ax.set_xlabel('Year')
ax.set_ylabel('Number of Patches')
ax.set_xticks(x)
ax.set_xticklabels(year)
ax.set_title('Pond')

# Create a second y-axis for area of presence
ax_2 = ax.twinx()
line2, = ax_2.plot(x, size_p, color='black', marker='o', label='Area of Presence')
ax_2.set_ylabel('Area of Presence (km²)')
_, ymax = ax_2.get_ylim()
ax_2.set_ylim(0, ymax)
# Combine legends
bars_legend = mpatches.Patch(facecolor = 'silver',edgecolor='black', label='Number of Patches')
lines_legend = Line2D([], [], color='black', marker='o', linestyle='-', label='Area of Presence')

legend = fig.legend(handles=[bars_legend, lines_legend], loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=12)
legend.get_frame().set_linewidth(0) 
plt.tight_layout()
filepath = 'D:\\OneDrive - Clark University\\Desktop\\Research\\patchmanuscript\\graphs\\'
plt.savefig(filepath + 'pond'+'netchange.tif', bbox_inches='tight',format='tif')



#%% gross change vs net change
sum_gross = df_patch_size_marsh.iloc[:,2:10].sum().sum()
sum_ggain = df_patch_size_marsh.iloc[:,2:6].sum().sum()
sum_gloss = df_patch_size_marsh.iloc[:,6:10].sum().sum()
net_sum = abs(sum_ggain - sum_gloss)
# gross versus net
gvn = sum_gross/net_sum
print(f"gross change in area is {gvn} bigger than net in marsh")
sum_gross = df_patch_size_pond.iloc[:,2:10].sum().sum()
sum_ggain = df_patch_size_pond.iloc[:,2:6].sum().sum()
sum_gloss = df_patch_size_pond.iloc[:,6:10].sum().sum()
net_sum = abs(sum_ggain - sum_gloss)
# gross versus net
gvn = sum_gross/net_sum

print(f"gross change in area is {gvn} bigger than net in pond")

#%% gross change vs net change
sum_gross = abs(df_inde_all_marsh.iloc[:,1:]).sum().sum()
sum_gin = (abs(df_inde_all_marsh.iloc[:,2])+abs(df_inde_all_marsh.iloc[:,3])).sum()
sum_gde = (abs(df_inde_all_marsh.iloc[:,1])+abs(df_inde_all_marsh.iloc[:,4])).sum()
net_sum = abs(sum_gin - sum_gde)
# gross versus net
gvn = sum_gross/net_sum
print(f"gross change in number is {gvn} bigger than net in marsh")
sum_gross = abs(df_inde_all_pond.iloc[:,1:]).sum().sum()
sum_gin = (abs(df_inde_all_pond.iloc[:,2])+abs(df_inde_all_pond.iloc[:,3])).sum()
sum_gde = (abs(df_inde_all_pond.iloc[:,1])+abs(df_inde_all_pond.iloc[:,4])).sum()
net_sum = abs(sum_gin - sum_gde)
# gross versus net
gvn = sum_gross/net_sum

print(f"gross change in number is {gvn} bigger than net in pond")

