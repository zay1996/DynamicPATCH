
# -*- coding: utf-8 -*-
"""
This script include all functions that compute and generate all resulting 
tables and charts from DynamicPATCH

@author: Aiyin Zhang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from dynamicpatch.config import cat_dict, df_cat, year, nt, res, connectivity


class Gen_Charts:
    def __init__(self, pattern, areaunit = None,type_ = 'change'):
        '''
        Initialize parameters needed for creating graphics 

        Input Parameters
        ----------
        pattern : 3d array
            Transition pattern map for all time intervals.
        type_ : String, optional
            Specify whether the map if for change analysis or comparison analysis. 
            The default is 'change'.
        areaunit : String, optional
            Unit of area. The default is 'km2'. Total of 3 options: 'sqm2','pixels',
            and 'km2'.
        
        Parameters initialized 
        -----------
        :df_patch_num : Dataframe
            Number of transition patches by transition type. 
        :df_patch_size : Dataframe
            Total area of each transition type. 
        :patch_ave : Dataframe
            Average area of transition patches by transition type.
        :patch_median : Dataframe
            Median area of transition patches by transition type.
        :patch_ave_q1 : Dataframe
            First quartile of area in transition patches by transition type.
        :patch_ave_q3 : Dataframe
            Third quartile of area in transition patches by transition type.
        '''
        self.df_patch_num = pd.DataFrame(columns = ['year'] + list(cat_dict.keys())[1:])    
        self.df_patch_size = pd.DataFrame(columns = ['year'] + list(cat_dict.keys())[1:])    
        self.patch_ave = pd.DataFrame(columns = ['year'] + list(cat_dict.keys())[1:])    
        self.patch_median = pd.DataFrame(columns = ['year'] + list(cat_dict.keys())[1:])    
        self.patch_ave_sem = pd.DataFrame(columns = ['year'] + list(cat_dict.keys())[1:])    
        self.patch_ave_q1 = pd.DataFrame(columns = ['year'] + list(cat_dict.keys())[1:])
        self.patch_ave_q3 = pd.DataFrame(columns = ['year'] + list(cat_dict.keys())[1:])
        self.pattern = pattern
        # automatically assign appropriate areaunit 
        size_map = len(str(np.size(pattern)*res))
        self.areaunit = areaunit
        if areaunit is None:
            if(res == 0):
                self.areaunit = 'pixels'
            if(size_map > 6):
                self.areaunit = 'km2'
            if(size_map <= 6 & size_map > 0):
                self.areaunit = 'sqm2'     
        print(f"areaunit = {self.areaunit},size_map = {size_map},res = {res}")
        self.type_ = type_
        
        self.countnumsize()
            
    def countnumsize(self):
        '''
        Compute general stats of transition patches - total area, median, average size, first 
        and third quartile, and number of patches. Breaking down by each transition
        type

        Returns
        -------
        df_patch_num : Dataframe
            Number of transition patches by transition type. 
        df_patch_size : Dataframe
            Total area of each transition type. 
        patch_ave : Dataframe
            Average area of transition patches by transition type.
        patch_median : Dataframe
            Median area of transition patches by transition type.
        patch_ave_q1 : Dataframe
            First quartile of area in transition patches by transition type.
        patch_ave_q3 : Dataframe
            Third quartile of area in transition patches by transition type.
    
        '''
        areaunit = self.areaunit
        interval = []
        for i,y in enumerate(year[0:-1]):
            interval.append(str(year[i])+'-' + str(year[i+1]))
            
        self.df_patch_num['year']=year[0:-1]
        self.df_patch_size['year']=year[0:-1]
        self.patch_ave['year'] = interval
        self.patch_ave_sem['year'] = interval 
        self.patch_ave_q1['year'] = interval 
        self.patch_ave_q3['year'] = interval
        self.patch_median['year'] = interval
        iter_var = year[0:-1]

    
        for i,y in enumerate(iter_var):
            interval.append(str(year[i])+'-' + str(year[i+1]))
            num_patch, size_patch, ave_patch,med_patch,sem_patch,q1_patch,q3_patch = [],[],[],[],[],[],[]
            for c in df_cat['Value'][1:]:
                tax_map = (self.pattern[i] == c).astype('uint8')
                num_labels, patchlabels = cv2.connectedComponents(tax_map, connectivity=connectivity)
                num_labels = num_labels - 1 # exclude background 
                num_patch.append(num_labels)
                #ave_patch.append(np.sum(tax_map)/num_labels)
                size_list = np.unique(patchlabels,return_counts=True)[1][1:]
                
                
                if(areaunit == 'pixels'):
                    size_list = size_list
                    size_patch.append(np.sum(tax_map))
                if(areaunit == 'sqm2'):
                    size_list = size_list * (res**2)
                    size_patch.append(np.sum(tax_map)* (res**2))
                if(areaunit == 'km2'):
                    size_list = [size * (res**2)/(1000**2) for size in size_list]
                    size_patch.append((np.sum(tax_map)* (res**2)/(1000**2)))
                
                ave_patch.append(np.mean(size_list))
                sem_patch.append(np.std(size_list)/np.sqrt(num_labels-1))
                med_patch.append(np.median(size_list)) ## get median instead
                #std_patch.append(np.std(size_list))
                if(len(size_list)>=1):
                    q1_patch.append(np.percentile(size_list, 25))
                    q3_patch.append(np.percentile(size_list, 75))
                else:
                    q1_patch.append(0)
                    q3_patch.append(0)
            self.df_patch_num.iloc[i,1:] = num_patch
            self.df_patch_size.iloc[i,1:] = size_patch
            self.patch_ave.iloc[i,1:] = ave_patch
            self.patch_ave_q1.iloc[i,1:] = q1_patch
            self.patch_ave_q3.iloc[i,1:] = q3_patch
            self.patch_median.iloc[i,1:] = med_patch
       


    def plot_ave_size(self,width = 0.35, ax = None,log_scale=True):
        '''
        Plot size distribution of transition patches. Include median, average, and IQR.
    
        Parameters
        ----------
        width : float, optional
            Width of bars. The default is 0.35.
        ax : axis, optional
            Figure axis, use when plot as a subfigure. The default is None.
        log_scale : Boolean, optional
            Whether log scale is applied to the y-axis. The default is False.
    
        Returns
        -------
        fig: Figure
            Stacked bar chart figure.
        title: String
            Title of the stacked bar chart.
    
        '''
        areaunit = self.areaunit
        if ax is None:
            fig, ax = plt.subplots(figsize=(10,6))
            flag_ax = False
    
        df_types = self.patch_median.iloc[:, 2:]
        
        # Setting the positions and width for the bars
        x = np.arange(len(df_types.columns))  # the label locations
         # the width of the bars
        
        colorlist = []
        for cat in df_types.columns:
            colors_ = df_cat.loc[df_cat['Type'] == cat, 'Color']
            for color in colors_:
                colorlist.append(color)
        # Calculate asymmetrical error
        
        iter_val = len(year[0:-1])
    
        for y in range(iter_val):
            q1 = self.patch_ave_q1.iloc[y,2:]
            q3 = self.patch_ave_q3.iloc[y,2:]
            yerr_lower = df_types.iloc[y] - q1
            yerr_upper = q3 - df_types.iloc[y]    
            bars = ax.bar(x - width*(len(df_types)/2)+y*width, df_types.iloc[y], \
                          width, yerr=[yerr_lower, yerr_upper], align = 'edge',label=self.patch_ave.year[y],\
                              edgecolor = 'black',color = colorlist)
            means = self.patch_ave.iloc[y, 2:].values
            for i, mean_val in enumerate(means):
                ax.scatter(x[i] - width*(len(df_types)/2)+y*width + width/2, mean_val, color=colorlist[i], edgecolor = 'black',marker='o', s=30, zorder=3)
            
    
            
        # Adding some text for labels, title and custom x-axis tick labels, etc.
        ax.set_xlabel('Transition Types',fontsize = 18)
        if (areaunit == 'pixels'):
            ax.set_ylabel('Size of transition patch (number of pixels)',fontsize = 14)
        if (areaunit == 'sqm2'):
            ax.set_ylabel('Size of transition patch (Square Meters)',fontsize = 14)
        if (areaunit == 'km2'):
            ax.set_ylabel('Size of transition patch (km²)',fontsize = 14)
        if log_scale:
            ax.set_yscale('log')    
        #ax.set_title('Bar chart by column names')
        ax.set_xticks(x)
        ax.set_xticklabels(df_types.columns,rotation = 45)
        # Creating the legend
        from matplotlib.lines import Line2D
        from matplotlib.patches import Rectangle
        # Creating legend elements
        legend_elements = [
            Line2D([0, 0], [0, 1], color='black',  marker='|', linestyle='None', markersize = 13, markeredgewidth=1.5, label='IQR'),
            Rectangle((0, 0), width = 1, height = 10, edgecolor='black', facecolor='none',  linewidth = 1.5, label='Median'),
            Line2D([0], [0], marker='o', color='black', linestyle='None', markersize=10, markerfacecolor='none', markeredgewidth=1.5, label='Mean')
        ]
        
    
        ax.legend(handles=legend_elements, handleheight = 2, handlelength = 1, loc='upper left',ncol = 3) 
        # Display the plot
        title = 'Distribution of transition patch sizes'
        fig.tight_layout()
        
        if flag_ax is False:
            #plt.show()   
            return fig, title
        else:
            return ax

    def plot_num(self,width = 0.35, ax = None):
     
        '''
        Plot number of transition patches by each transition type    
    
        Parameters
        ----------
        width : float, optional
            Width of bars. The default is 0.35.
        ax : axis, optional
            Figure axis, use when plot as a subfigure. The default is None.
    
        Returns
        -------
        fig: Figure
            Stacked bar chart figure.
        title: String
            Title of the stacked bar chart.
    
        '''   
        if ax is None:
            fig, ax = plt.subplots(figsize=(10,6))
            flag_ax = False
    
        df_types = self.df_patch_num.iloc[:, 2:]
        
        # Setting the positions and width for the bars
        x = np.arange(len(df_types.columns))  # the label locations
        #width = 0.35  # the width of the bars
        
        colorlist = []
        for cat in df_types.columns:
            colors_ = df_cat.loc[df_cat['Type'] == cat, 'Color']
            for color in colors_:
                colorlist.append(color)
        if(self.type_ == 'change'):
            iter_var = year[0:-1]
        if(self.type_ == 'compare'):     
            iter_var = year
        for y in range(len(iter_var)):
            bars = ax.bar(x - width*(len(df_types)/2)+y*width, df_types.iloc[y], \
                          width, align = 'edge',label=self.df_patch_num.year[y],\
                              edgecolor = 'white',color = colorlist)
        
        # Adding some text for labels, title and custom x-axis tick labels, etc.
        ax.set_xlabel('Transition Types',fontsize = 18)
        ax.set_ylabel('Number of Transition Patches', fontsize = 14)
        #ax.set_title('Bar chart by column names')
        ax.set_xticks(x)
        ax.set_xticklabels(df_types.columns,rotation = 45)
        #ax.legend()
        title = 'Number of transition patch for each transition type'
        fig.tight_layout()
        if flag_ax is False:
        # Display the plot
            #plt.show()    
            return fig,title
        else:
            return ax
        
    def inde_table(self,df_inde):    
        '''
        Create table for annual gross increase and gross decrease in number 
        of pathces. Used as input data for function inde_stackedbars()
    
        Parameters
        ----------
        df_inde : Dataframe
        Dataframe with gross increase and gross decrease due to four count
        altering transition types.

        Returns
        -------
        df_inde : Dataframe
        Dataframe with gross increase and gross decrease due to four count
        altering transition types.
        df_indey : Dataframe
        Dataframe with ANNUAL gross increase and gross decrease due to four count
        altering transition types.
        inline : float
            Increase Line.
        deline : float
            Decrease Line.
    
        '''
        ########## CALCULATE GROSS INCREASE AND DECREASE OF EACH TYPE
        #grossin
        grossin = df_inde['Splitting'] + df_inde['Appearing']
        #grossde
        grossde = df_inde['Disappearing'] + df_inde['Merging']
        
        # increased line
        inline = np.sum(grossin)/(year[-1]-year[0])
        deline = np.sum(grossde)/(year[-1]-year[0])
        
        
        ## Do dpt, spt, apt, cpt need to be annual? 
        df_indey=pd.DataFrame(columns = ['year','Disappearing','Appearing','Splitting','Merging'])
        if(self.type_ == 'change'):
            df_indey['year']=year[0:-1]
            df_indey.iloc[:,1:] = df_inde.iloc[:,1:].div(np.diff(year),axis = 0)
        if(self.type_ == 'compare'):
            df_indey['year'] = year
            df_indey.iloc[:,1:] = df_inde.iloc[:,1:]
    
        return df_inde,df_indey,inline,deline
        # compute N
        
    def inde_stackedbars(self,df_inde,ax = None,legend = 'Yes'):
        '''
        Create stacked bars for annual gross increase and gross decrease in 
        number of patches
    
        Parameters
        ----------
        df_inde : Dataframe
        Dataframe with gross increase and gross decrease due to four count
        altering transition types.
        ax : axis, optional
            Figure axis, use when plot as a subfigure. The default is None.
        legend : String, optional
            Whether a legend is needed. The default is 'Yes'.
    
        Returns
        -------
        fig: Figure
            Stacked bar chart figure.
        title: String
            Title of the stacked bar chart.
    
        '''
        df_inde,df_indey,inline,deline = self.inde_table(df_inde)
        title = 'Annual Gross Increase and Decrease in Number of Patches'
        if ax is None:
        ### PLOT INCREASE AND DECREASE ####
            fig, ax = plt.subplots(figsize=(10,6))
            ax_flag = False
        
        
        xlabels=np.array(year)
        
        if(self.type_=='change'):
            width=np.diff(year)
        if(self.type_ == 'compare'):
            width = 0.7
            x = np.arange(len(xlabels))
            
        colorlist = []
        for cat in df_indey.columns[1:]:
            colorlist.append(df_cat[df_cat['Type'] == cat]['Color'])
            
            
        #colorlist = ['#993404','#0570b0','orange','#00A9E6']
        bargheight = df_indey.copy()
        bargheight['Splitting'] = df_indey['Appearing']
        bargheight['Merging'] = df_indey['Disappearing']
        bargheight['Disappearing'] = 0
        bargheight['Appearing'] = 0
        #legendlist = ['Disappearance','Split','Appearance','Coalescence']
        legendlist = df_indey.columns[1:]
        
        
        p=[]
        if(self.type_ == 'change'):
            for i in range(4):
                #print(i+1,df_indey.iloc[:,i+1],width,bargheight.iloc[:,i+1],colorlist[i])
                p.append(ax.bar(df_indey['year'],df_indey.iloc[:,i+1],width=width,bottom=bargheight.iloc[:,i+1], color = colorlist[i], align='edge', label = legendlist[i]))
            ax.set_xlabel('Time Interval',fontsize=14)
            print(xlabels)
            ax.set_xticks(xlabels.astype(int))  # Set the positions of the ticks
            ax.set_xticklabels(xlabels)         # Set the labels for the ticks  
            ax.set_ylabel('Annual decrease and increase (number of patches)',fontsize=12)
        if(self.type_ == 'compare'):    
            for i in range(4):
                #print(i+1,df_indey.iloc[:,i+1],width,bargheight.iloc[:,i+1],colorlist[i])
                p.append(ax.bar(x,df_indey.iloc[:,i+1],width=width,bottom=bargheight.iloc[:,i+1], color = colorlist[i], align='center', label = legendlist[i]))
            
            ax.set_xlabel('Time',fontsize = 14)
            ax.set_ylabel('Decrease and increase (number of patches)',fontsize=12)
            plt.xticks(x,xlabels)

        ax.axhline(y=0,color='0',linewidth=0.5)
        
        ax.axhline(y=inline, color = 'black', linewidth = 2, label = 'Increase Line', linestyle = 'dashed')
        ax.axhline(y=deline, color = 'black', linewidth = 2, label = 'Decrease Line', linestyle = 'dashdot')

        # Adjust legend order
        if(legend == 'Yes'):
            handles, labels = ax.get_legend_handles_labels()
            new_order = [0, 4, 3, 1, 2, 5]  # Order to display in legend
            ax.legend([handles[i] for i in new_order], [labels[i] for i in new_order], ncol=1, bbox_to_anchor=(1.1, 0.5), loc='lower center')        
        
        fig.tight_layout()
        if ax_flag is False:
            #plt.show()
            return fig, title
        else:
            return ax
    
        
    def gainloss_table(self, option = 'area'):
        '''
        Create table for annual gross gain and gross loss by transition type. U
        sed as input data for function gainloss_stackedbars()    
    
        Parameters
        ----------
        option : String, optional
            Option for the unit of the y-axis. The default is 'area'.

    
        Returns
        -------
        dfbarsize : Dataframe
            Annual gross change by transition type
        gainline : Float
            Gain Line
        lossline : Float
            Lossline 
    
        '''
        
        
        areaunit = self.areaunit
        pattern = self.pattern
        
        # compute union presence v
        V = np.sum(pattern>0)/(year[-1] - year[0]) # in number of pixels 
            
        
        dfbarsize=pd.DataFrame(columns =  ['year','Appearing','Merging', 'Filling','Expanding','Disappearing','Splitting','Perforating','Contracting'])
        
        
        if(self.type_ == 'change'):
            dfbarsize['year']=year[0:-1]
        if(self.type_ == 'compare'):
            dfbarsize['year'] = year
        df_patch_option = self.df_patch_size.copy()
        if(option == 'percentage'):
            df_patch_option.iloc[:,1:] = self.df_patch_size.iloc[:,1:]/V
        if (option == 'area' and areaunit == 'pixels'):
            df_patch_option.iloc[:,1:] = self.df_patch_size.iloc[:,1:]
        for c in dfbarsize.columns[1:]:
            #print(c)
            if(self.type_ == 'change'):
                dfbarsize[c] = df_patch_option[c]/np.diff(year)
            if(self.type_ == 'compare'):
                dfbarsize[c] = df_patch_option[c]
                
        n = len(dfbarsize.columns)-1
        dfbarsize.iloc[:,int(n/2)+1:] = dfbarsize.iloc[:,int(n/2)+1:]*-1
    
            
        # gain and loss line
        gainline = sum(df_patch_option['Expanding'] + df_patch_option['Filling'] + df_patch_option['Merging'] + df_patch_option['Appearing'])/(year[-1]-year[0])
        lossline = -1*sum(df_patch_option['Contracting'] + df_patch_option['Perforating'] + df_patch_option['Splitting'] + df_patch_option['Disappearing'])/(year[-1]-year[0])
        
        return dfbarsize,gainline,lossline
    
    def gainloss_stackedbars(self, option = 'area',ax = None,legend = 'yes'):
        '''
        Plot the annual gains and losses of each transition type 
    
    
        Parameters
        ----------
        option : String, optional
            Option for the unit of the y-axis. The default is 'area'.
        ax : axis, optional
            Figure axis, use when plot as a subfigure. The default is None.
        legend : String, optional
            Whether a legend is needed. The default is 'Yes'.

        Returns
        -------
        fig: Figure
            Stacked bar chart figure.
        title: String
            Title of the stacked bar chart.
        '''    
        type_ = self.type_
        areaunit = self.areaunit
        dfbarsize,gainline,lossline = self.gainloss_table()
    
    
        n = len(dfbarsize.columns)-1
        ### PLOT LOSS AND GAIN SIZES ####
        if ax is None:
            fig, ax = plt.subplots(figsize=(10,6))
            ax_flag = False
        
        
        xlabels=np.array(year).astype('str')
        if (type_ == 'compare'):
            width = 0.7
            x = np.arange(len(xlabels)) 
            
        if (type_ == 'change'):
            width=np.diff(year)
            
        colorlist = []
        for cat in dfbarsize.columns[1:]:
            colorlist.append(df_cat[df_cat['Type'] == cat]['Color'])
                
        
        legendlist = dfbarsize.columns[1:]
        
        dflossbottom = dfbarsize.iloc[:,int(n/2)+1:].cumsum(axis=1)
        dfgainbottom = dfbarsize.iloc[:,1:int(n/2)+1].cumsum(axis=1)
        dfgainbottom.insert(0, "0", np.zeros(nt))
        dflossbottom.insert(0,"0",np.zeros(nt))
        
    
        p=[]
        
        if(type_ == 'change'):
            for i in range(len(dfbarsize.columns)-1):
                #print(i)
                if (i < n/2):
                    p.append(ax.bar(dfbarsize['year'],dfbarsize.iloc[:,i+1],width=width,bottom=dfgainbottom.iloc[:,i], color = colorlist[i], align='edge', label = legendlist[i]))
                if (i >= n/2):
                    p.append(ax.bar(dfbarsize['year'],dfbarsize.iloc[:,i+1],width=width,bottom=dflossbottom.iloc[:,i-int(n/2)], color = colorlist[i], align='edge', label = legendlist[i]))
            ax.set_xlabel('Time Interval',fontsize=14)
            ax.set_xticks(xlabels.astype(int))  # Set the positions of the ticks
            ax.set_xticklabels(xlabels)         # Set the labels for the ticks
                
        if(type_ == 'compare'):
            for i in range(len(dfbarsize.columns)-1):
                #print(i)
                if (i < n/2):
                    p.append(ax.bar(x,dfbarsize.iloc[:,i+1],width=width,bottom=dfgainbottom.iloc[:,i], color = colorlist[i], align='center', label = legendlist[i]))
                if (i >= n/2):
                    p.append(ax.bar(x,dfbarsize.iloc[:,i+1],width=width,bottom=dflossbottom.iloc[:,i-int(n/2)], color = colorlist[i], align='center', label = legendlist[i]))
            ax.set_xlabel('Time',fontsize = 14)       
            plt.xticks(x,xlabels)
                       
        if(option == 'percentage'):
            if(type_ == 'change'):
                ax.set_ylabel("Annual loss and gain (% out of Union Presence)",fontsize = 14)
            if(type_ == 'compare'):
                ax.set_ylabel("Loss and gain (% out of Union Presence)",fontsize = 14)
        if(option == 'area'):
            if(type_ == 'change' and areaunit == 'pixels'):
                ax.set_ylabel('Annual loss and gain (number of pixels)',fontsize=14)
            if(type_ == 'change' and areaunit == 'sqm2'):
                ax.set_ylabel('Annual loss and gain (Square Meters)',fontsize = 14)
            if(type_ == 'change' and areaunit == 'km2'):
                ax.set_ylabel('Annual loss and gain (km²)',fontsize = 14)
            if(type_ == 'compare'):
                ax.set_ylabel("Loss and gain (number of pixels)",fontsize = 14)
        #plt.ylim(-2,2)
        ax.axhline(y=0,color='0',linewidth=0.5)
        
        
        ax.axhline(y=gainline, color = 'black', linewidth = 2, label = 'Gain Line', linestyle = 'dashed')
        ax.axhline(y=lossline, color = 'black', linewidth = 2, label = 'Loss Line', linestyle = 'dashdot')
        #plt.legend((p[0][0], p[1][0],p[2][0],p[3][0]), ('Disappearance','Split','Appearance','Coalescence')) 
        if(legend == 'yes'):
            handles, labels = ax.get_legend_handles_labels()
            new_order = [0, 5, 4, 3, 2, 1,6,7,8,9]  # Order to display in legend
            ax.legend([handles[i] for i in new_order], [labels[i] for i in new_order], ncol=1, bbox_to_anchor=(1.1, 0.35), loc='lower center')
    
        fig.tight_layout()
        title = 'Annual Gross Loss and Gross Gain by Transition Types'
        
        if ax_flag is False:
            #plt.show()
            return fig,title
        else:
            return ax        
