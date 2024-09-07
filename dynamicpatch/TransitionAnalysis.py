# -*- coding: utf-8 -*-
"""
This script identifies transition types from an input binary classification at 
t0 and t1

@author: AiZhang
"""
from osgeo import gdal
import numpy as np
import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
    
from skimage.morphology import erosion, dilation
from skimage.morphology import disk
from skimage.morphology import square

from dynamicpatch.config import cat_dict

class TransitionAnalysis:
    def __init__(self,params,classt0,classt1,year):
        self.absence, self.presence, self.nodata, self.nt, self.nl, self.ns, \
            self.connectivity = params
        self.classt0 = classt0
        self.classt1 = classt1
        self.year = year
        self.binaryclass = np.zeros((2, self.nl, self.ns), dtype='uint8')
        self.binaryclass[0][self.classt0 == self.presence] = 1
        self.binaryclass[1][self.classt1 == self.presence] = 1
        self.setup_data()        
    def setup_data(self):
        self.datacrosst = np.zeros((self.nl, self.ns), dtype='byte')
        for i in range(len(self.year)):
            self.datacrosst[(self.binaryclass[0] == 0) & (self.binaryclass[1] == 0)] = 1
            self.datacrosst[(self.binaryclass[0] == 1) & (self.binaryclass[1] == 1)] = 2
            self.datacrosst[(self.binaryclass[0] == 0) & (self.binaryclass[1] == 1)] = 3
            self.datacrosst[(self.binaryclass[0] == 1) & (self.binaryclass[1] == 0)] = 4

        self.gain = np.zeros((self.nl, self.ns), dtype='uint8')
        self.gain[(self.datacrosst == 3)] = 1

        self.loss = np.zeros((self.nl, self.ns), dtype='uint8')
        self.loss[(self.datacrosst == 4)] = 1

        self.persmap = np.zeros((self.nl, self.ns), dtype='uint8')
        self.persmap[(self.datacrosst == 2)] = 1

        self.absmap = np.zeros((self.nl, self.ns), dtype='uint8')
        self.absmap[(self.datacrosst == 1)] = 1

        self.gainpatchlabels, self.losspatchlabels, self.perpatchlabels = \
            [np.zeros((self.nl, self.ns), dtype='int') for _ in range(3)]
        self.prepatchlabelst0, self.prepatchlabelst1, self.upatch = \
            [np.zeros((self.nl, self.ns), dtype='int') for _ in range(3)]
        self.label_patches()
    
        
    def label_patches(self):
        _, self.gainpatchlabels = cv2.connectedComponents(self.gain, connectivity=self.connectivity)
        _, self.losspatchlabels = cv2.connectedComponents(self.loss, connectivity=self.connectivity)
        _, self.perpatchlabels = cv2.connectedComponents(self.persmap, connectivity=self.connectivity)
        _, self.prepatchlabelst0 = cv2.connectedComponents(self.binaryclass[0], connectivity=self.connectivity)
        _, self.prepatchlabelst1 = cv2.connectedComponents(self.binaryclass[1], connectivity=self.connectivity)
        _, self.upatch = cv2.connectedComponents((self.datacrosst > 1).astype('uint8'), connectivity=self.connectivity)
        
        
        absencet0 = (1 - self.binaryclass[0]).astype('uint8')
        _, abslabelt0 = cv2.connectedComponents(absencet0, connectivity=self.connectivity)

        absencet1 = (1 - self.binaryclass[1]).astype('uint8')
        _, abslabelt1 = cv2.connectedComponents(absencet1, connectivity=self.connectivity)

        self.abslabelt0 = abslabelt0
        self.abslabelt1 = abslabelt1
        return abslabelt0, abslabelt1

    def dilate(self, patch):
        if self.connectivity == 8:
            return dilation(patch, square(3))
        elif self.connectivity == 4:
            dilate4conn = np.zeros((3, 3)).astype('byte')
            dilate4conn[1, :] = 1
            dilate4conn[0, 1] = 1
            dilate4conn[2, 1] = 1
            return dilation(patch, dilate4conn)
    
    def identify(self):
        
        pattern = np.zeros((self.nl, self.ns), dtype='int')
        
        border_nodata = np.zeros((self.nl, self.ns), dtype='ubyte')
        border_nodata[[0, -1], :] = 1
        border_nodata[:, [0, -1]] = 1
        
        nodata_ar = np.zeros((self.nl, self.ns), dtype='ubyte')
        nodata_ar[self.classt0 == self.nodata] = 1
        nodata_ar_di = self.dilate(nodata_ar)
        border_nodata[nodata_ar_di == 1] = 1

        ext_abs_t0 = self.abslabelt0 * border_nodata
        ind_ext_abs_t0 = np.unique(ext_abs_t0)
        ex_abs_labelt0 = self.abslabelt0 * (np.isin(self.abslabelt0, ind_ext_abs_t0))

        ext_abs_t1 = self.abslabelt1 * border_nodata
        ind_ext_abs_t1 = np.unique(ext_abs_t1)
        ex_abs_labelt1 = self.abslabelt1 * (np.isin(self.abslabelt1, ind_ext_abs_t1))

        
        lossbi = self.loss.astype(int)
        gainbi = self.gain.astype(int)
        
    
        dilated_loss = self.dilate(lossbi)
        edge_loss=dilated_loss-lossbi
        
        dilated_gain = self.dilate(gainbi)
        edge_gain=dilated_gain-gainbi
        
        ##### IDENTIFY CONNECTION WITH GAIN PATCHES AT TIME POINTS
        
        ## Initial Presence
        ed_ip = edge_gain * self.prepatchlabelst0 # Initial patch labels in the dilated edge_gain
        ed_labels = edge_gain * self.dilate(self.gainpatchlabels) # loss patch labels in the dilated edge_gain 
        # using dilated labels could cause a problem cause if two gain patch only has 1 pixel distance then the patch with the larger index can override the other. 
        
        # the goal is to count the number of Initial patches for each loss patch 
        flat_A = ed_ip[ed_labels>0].ravel() # only look at where there are loss patches outer edge_gain
        flat_B = ed_labels[ed_labels>0].ravel() 
        
        df = pd.DataFrame({'A': flat_A, 'B': flat_B})
        df['A'] = df['A'].replace(0, np.nan) # replace 0 with nan so we don't count 0 as one patch 
        g_ip_counts = df.groupby('B')['A'].nunique().fillna(0)
        
        ## Final Presence 
        ed_fp = edge_gain * self.prepatchlabelst1 # Final patch labels in the dilated edge_gain
        
        # the goal is to count the number of initial patches for each loss patch 
        flat_A = ed_fp[ed_labels>0].ravel() # only look at where there are loss patches outer edge_gain
        flat_B = ed_labels[ed_labels>0].ravel() 
        
        df = pd.DataFrame({'A': flat_A, 'B': flat_B})
        df['A'] = df['A'].replace(0, np.nan) # replace 0 with nan so we don't count 0 as one patch 
        g_fp_counts = df.groupby('B')['A'].nunique().fillna(0)    
    
        ## Initial External Absence
        ed_iea = edge_gain * ex_abs_labelt0
        
        # try instead of using the dilated edge itself, dilate the edge inward and get the inner edge so that it matches the original labels
        ed_iea_di = self.dilate(ed_iea)
        in_iea = ed_iea_di * gainbi 
        
        #flat_A = ed_iea[ed_labels>0].ravel()
        flat_A = in_iea[self.gainpatchlabels>0].ravel()
        flat_B = self.gainpatchlabels[self.gainpatchlabels>0].ravel()
        df = pd.DataFrame({'A': flat_A, 'B': flat_B})
        df['A'] = df['A'].replace(0, np.nan) # replace 0 with nan so we don't count 0 as one patch 
        g_iea_counts = df.groupby('B')['A'].nunique().fillna(0)    
    
        ## Final Absence
        ed_fea = edge_gain * ex_abs_labelt1
        
        ed_fea_di = self.dilate(ed_fea)
        in_fea = ed_fea_di * gainbi
        flat_A = in_fea[self.gainpatchlabels>0].ravel()
        #flat_A = ed_fea[ed_labels>0].ravel()
        #flat_B = ed_labels[ed_labels>0].ravel() 
        
        df = pd.DataFrame({'A': flat_A, 'B': flat_B})
        df['A'] = df['A'].replace(0, np.nan) # replace 0 with nan so we don't count 0 as one patch 
        g_fea_counts = df.groupby('B')['A'].nunique().fillna(0)       
        
        
        ##### IDENTIFY CONNECTION WITH LOSS PATCHES AT TIME POINTS
        
        ## Initial Presence
        ed_ip = edge_loss * self.prepatchlabelst0 # Initial patch labels in the dilated edge_loss
        ed_labels = edge_loss * self.dilate(self.losspatchlabels) # loss patch labels in the dilated edge_loss 
        
        # the goal is to count the number of Initial patches for each loss patch 
        flat_A = ed_ip[ed_labels>0].ravel() # only look at where there are loss patches outer edge_loss
        flat_B = ed_labels[ed_labels>0].ravel() 
        
        df = pd.DataFrame({'A': flat_A, 'B': flat_B})
        df['A'] = df['A'].replace(0, np.nan) # replace 0 with nan so we don't count 0 as one patch 
        l_ip_counts = df.groupby('B')['A'].nunique().fillna(0)
        
        ## Final Presence 
        ed_fp = edge_loss * self.prepatchlabelst1 # Final patch labels in the dilated edge_loss
        
        # the goal is to count the number of initial patches for each loss patch 
        flat_A = ed_fp[ed_labels>0].ravel() # only look at where there are loss patches outer edge_loss
        flat_B = ed_labels[ed_labels>0].ravel() 
        
        df = pd.DataFrame({'A': flat_A, 'B': flat_B})
        df['A'] = df['A'].replace(0, np.nan) # replace 0 with nan so we don't count 0 as one patch 
        l_fp_counts = df.groupby('B')['A'].nunique().fillna(0)    
    
        ## Initial External Absence
        ed_iea = edge_loss * ex_abs_labelt0
        
        flat_A = ed_iea[ed_labels>0].ravel()
        
        df = pd.DataFrame({'A': flat_A, 'B': flat_B})
        df['A'] = df['A'].replace(0, np.nan) # replace 0 with nan so we don't count 0 as one patch 
        l_iea_counts = df.groupby('B')['A'].nunique().fillna(0)    
    
        ## Final Absence
        ed_fea = edge_loss * ex_abs_labelt1
        
        flat_A = ed_fea[ed_labels>0].ravel()
        
        df = pd.DataFrame({'A': flat_A, 'B': flat_B})
        df['A'] = df['A'].replace(0, np.nan) # replace 0 with nan so we don't count 0 as one patch 
        l_fea_counts = df.groupby('B')['A'].nunique().fillna(0)        
        
        ##### IDENTIFICATION OF TYPOLOGY ##### 
        ### APPEARING ###
        # new def: gain patch connect to zero final presence 
        app_ind = list(g_fp_counts[g_fp_counts == 0].index)
        pattern[np.isin(self.gainpatchlabels,app_ind)] = 1     
        ### MERGING ###
        ## new def: gain patch connect to at least (or exactly?) one final patch and more than one initial patch 
        merg_ind = list(np.intersect1d(list(g_ip_counts[g_ip_counts > 1].index), \
                                       list(g_fp_counts[g_fp_counts == 1].index)))
        pattern[np.isin(self.gainpatchlabels, merg_ind)] = 2 
        ### FILLING ###
        ## new def: connect to one initial presence and at least (or exactly?) one final presence, and connect to 0 final (?) exterior absence
        fill_ind = list(np.intersect1d(np.intersect1d(list(g_ip_counts[g_ip_counts == 1].index),\
                                               list(g_fp_counts[g_fp_counts >=1].index)),\
                                 list(g_fea_counts[g_fea_counts == 0].index)))
        pattern[np.isin(self.gainpatchlabels,fill_ind)] = 3 
        ### EXPANDING 
        ## new def: connect to one initial presence and at least (or exactly?) one final presence, and connect to 0 final exterior absence
        exp_ind = list(np.intersect1d(np.intersect1d(list(g_ip_counts[g_ip_counts == 1].index),\
                                               list(g_fp_counts[g_fp_counts >=1].index)),\
                                 list(g_fea_counts[g_fea_counts > 0].index)))
        pattern[np.isin(self.gainpatchlabels,exp_ind)] = 4
        
        ### DISAPPEARING ###
        ## new def: loss patch connect to zero initial presence
        dis_ind = list(l_ip_counts[l_ip_counts == 0].index)
        pattern[np.isin(self.losspatchlabels,dis_ind)] = 5 
        
        ### SPLITTING ###
        ## new def: loss patch connect to one initial presence and more than one final presence 
        split_ind = list(np.intersect1d(list(l_ip_counts[l_ip_counts == 1].index), \
                                        list(l_fp_counts[l_fp_counts > 1].index)))
        pattern[np.isin(self.losspatchlabels, split_ind)] = 6
        
        ### PERFORATING ###
        ## new def: connect to one final presence and [one initial presence], and connect to 0 initial (?) exterior absence 
        perf_ind = list(np.intersect1d(np.intersect1d(list(l_ip_counts[l_ip_counts == 1].index),\
                                               list(l_fp_counts[l_fp_counts ==1].index)),\
                                 list(l_iea_counts[l_iea_counts == 0].index)))
        pattern[np.isin(self.losspatchlabels,perf_ind)] = 7 
        
        ### CONTRACTING ###
        ## new def: connect to one final presence and one intial patch, and connect to initial exterior absence 
        cont_ind = list(np.intersect1d(np.intersect1d(list(l_ip_counts[l_ip_counts == 1].index),\
                                               list(l_fp_counts[l_fp_counts ==1].index)),\
                                 list(l_fea_counts[l_iea_counts > 0].index)))
        
        ## NOTE: Perhaps by definition, loss patch cannot be connected to more than one intial presence patch♦
    
        pattern[np.isin(self.losspatchlabels,cont_ind)] = 8 
        
        
        # assign persistence 
        pattern[self.persmap==1]=9
        # assign no value
        pattern[self.classt0 == self.nodata] = -1
        
        self.pattern = pattern        
        
        return pattern
    
    def gross_change(self):
        '''
        Compute gross change in number of patches break down by four count-altering 
        transition types

        Returns
        -------
        df_inde : dataframe
            Dataframe with gross increase and gross decrease due to four count
            altering transition types.

        '''
        prepatchlabelst0 = self.prepatchlabelst0
        prepatchlabelst1 = self.prepatchlabelst1
        pattern = self.pattern
        connectivity = self.connectivity
        absence = self.absence
        upatch = self.upatch
        persmap = self.persmap
        
        
        nb = self.nt+1
        nl,ns = self.nl, self.ns
        
        ## Identify patches
        unionsplitp = np.zeros((nl,ns))
        unionmergep = np.zeros((nl,ns))
        app_p = np.zeros((nl,ns),dtype='int')
        dis_p = np.zeros((nl,ns),dtype='int')
        split_ind = cat_dict['Splitting']
        merge_ind = cat_dict['Merging']
        dis_ind = cat_dict['Disappearing']   # consider moving this part to the start of the patch indentification code 
        app_ind = cat_dict['Appearing']
        
        
        unionsplit = (prepatchlabelst1 > 0) & ~(pattern == app_ind) |  (pattern == split_ind).astype('uint8') # not including appearance at t1
        unionmerge = (prepatchlabelst0 > 0) & ~(pattern == dis_ind) |  (pattern == merge_ind).astype('uint8') # not including disappearance at t0
        
    
        num_labels, unionsplitp = cv2.connectedComponents(unionsplit, connectivity=connectivity)
        num_labels, unionmergep = cv2.connectedComponents(unionmerge, connectivity=connectivity)
        num_labels, app_p = cv2.connectedComponents((pattern == app_ind).astype('uint8'), connectivity=connectivity)
        num_labels, dis_p = cv2.connectedComponents((pattern == dis_ind).astype('uint8'), connectivity=connectivity)
    
    
    
        upatchinc = np.zeros((nl,ns)).astype('int')
        upatchinc[upatchinc == 0] = 255
        upatchdec = np.zeros((nb-1,nl,ns)).astype('int')
        upatchdec[upatchdec == 0] = 255
    
        
        upatch = upatch.astype('int')
    
    
        t0patchnumlist,t1patchnumlist,t0patchsize,t1patchsize,\
            grossinclist,grossdeclist,persperc,upatchsize,\
                upatchchangesize,upatchchange = [],[],[],[],[],[],[],[],[],[]
        
        upatchsize = np.unique(upatch,return_counts = True)[1][1:]
        
        ### T0
        # Flatten the arrays
        flat_A = prepatchlabelst0[upatch!=0].ravel()
        flat_B = upatch[upatch!=0].ravel()
        
        # Create a DataFrame from the flattened arrays
        df = pd.DataFrame({'A': flat_A, 'B': flat_B})
        df['A'] = df['A'].replace(0, np.nan)
        
        # Group by the values in B and count the unique values in A for each group
        unique_counts = df.groupby('B')['A'].nunique()    
        
        t0patchnum = unique_counts 
        t0patchnumlist = t0patchnum
        # now flatten the unlabeled for size computation
        flat_A = self.classt0[upatch!=0].ravel()
        flat_B = upatch[upatch!=0].ravel()
        
        df = pd.DataFrame({'A': flat_A, 'B': flat_B})
        df['A'] = df['A'].replace(absence, np.nan) # set absence as nan to avoid counting absence as a patch
        unique_sum = df.groupby('B')['A'].sum()   
        t0patchsize = unique_sum 
        
        ### T1
        # Flatten the arrays
        flat_A = prepatchlabelst1[upatch!=0].ravel()
        flat_B = upatch[upatch!=0].ravel()
        
        # Create a DataFrame from the flattened arrays
        df = pd.DataFrame({'A': flat_A, 'B': flat_B})
        df['A'] = df['A'].replace(0, np.nan)
        # Group by the values in B and count the unique values in A for each group
        unique_counts = df.groupby('B')['A'].nunique().fillna(0)    
        
        t1patchnum = unique_counts
        t1patchnumlist = t1patchnum
        
        # get net change in each union patch for histogram
        upatchchange = t1patchnum - t0patchnum 
        
        # now flatten the unlabeled for size computation
        flat_A = self.classt1[upatch!=0].ravel()
        flat_B = upatch[upatch!=0].ravel()
        
        df = pd.DataFrame({'A': flat_A, 'B': flat_B})
        df['A'] = df['A'].replace(absence, np.nan)
        unique_sum = df.groupby('B')['A'].sum()   
        t1patchsize = unique_sum 
    
        ### compute percentage of persistence in each union patch 
        flat_A = persmap[upatch!=0].ravel()
        flat_B = upatch[upatch!=0].ravel()       
        df = pd.DataFrame({'A': flat_A, 'B': flat_B}) 
        unique_sum = df.groupby('B')['A'].sum()/df.groupby('B')['A'].count()
        persperc = unique_sum
                
        
        ##### COMPUTE INCREASE AND DECREASE
        ### DISAPPEARANCE
        flat_A = dis_p[upatch!=0].ravel()
        flat_B = upatch[upatch!=0].ravel()
        
        df = pd.DataFrame({'A': flat_A, 'B': flat_B})
        df['A'] = df['A'].replace(0, np.nan) # replace 0 with nan so we don't count 0 in
        unique_counts = df.groupby('B')['A'].nunique().fillna(0)  
        disnumlist = unique_counts 
        disnum = disnumlist.sum()
        
        ### APPEARANCE 
        flat_A = app_p[upatch!=0].ravel()
        flat_B = upatch[upatch!=0].ravel()
        
        df = pd.DataFrame({'A': flat_A, 'B': flat_B})
        df['A'] = df['A'].replace(0, np.nan)
    
        unique_counts = df.groupby('B')['A'].nunique().fillna(0)
        appnumlist = unique_counts
        appnum = unique_counts.sum() 
        
        #### SPLIT 
        flat_A = unionsplitp[upatch!=0].ravel()
        df = pd.DataFrame({'A': flat_A, 'B': flat_B})
        df['A'] = df['A'].replace(0, np.nan)
        unique_counts = df.groupby('B')['A'].nunique()
        splnumlist = t1patchnum - appnumlist - unique_counts
        splnum = splnumlist.sum()
        #### MERGE
        flat_A = unionmergep[upatch!=0].ravel()
        df = pd.DataFrame({'A': flat_A, 'B': flat_B})
        df['A'] = df['A'].replace(0, np.nan)
        unique_counts = df.groupby('B')['A'].nunique()
        mernumlist = unique_counts - t0patchnum + disnumlist  
        mernum = mernumlist.sum()
    
        ## give loss stats negative values
        disnum = disnum * -1
    
        grossinclist = appnumlist + splnumlist 
        grossdeclist = disnumlist + mernumlist
        
        
        indices = ['Disappearing', 'Appearing', 'Splitting', 'Merging']
        
        df_inde  = pd.Series(index=indices,dtype='float64')
        
        df_inde['Disappearing'] = disnum
        df_inde['Appearing'] = appnum
        df_inde['Splitting'] = splnum
        df_inde['Merging'] = mernum
        
        
        return df_inde    
    
