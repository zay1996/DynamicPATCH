# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 21:33:38 2024

@author: aiiyi
"""

from osgeo import gdal
import numpy as np

    
def writedata(FileName,image,data1,type_):
    nb=np.shape(image)[0]
    driver=data1.GetDriver()    

    if(image.ndim==3):
        nl = np.shape(image)[1]
        ns = np.shape(image)[2]
        if(type_=='byte'):
            outData=driver.Create(FileName,ns,nl,nb,gdal.GDT_Byte)   
        if(type_=='int'):
            outData=driver.Create(FileName,ns,nl,nb,gdal.GDT_Int32)    
        if(type_=='float'):
            outData=driver.Create(FileName,ns,nl,nb,gdal.GDT_Float64)    
    if(image.ndim==2):
        nl = np.shape(image)[0]
        ns = np.shape(image)[1]
        if(type_=='byte'):
            outData=driver.Create(FileName,ns,nl,1,gdal.GDT_Byte)   
        if(type_=='int'):
            outData=driver.Create(FileName,ns,nl,1,gdal.GDT_Int32)    
        if(type_=='float'):
            outData=driver.Create(FileName,ns,nl,1,gdal.GDT_Float64)    
    
    geotrans=data1.GetGeoTransform()
    outData.SetGeoTransform(geotrans)
    proj=data1.GetProjection()
    outData.SetProjection(proj)
    if(image.ndim==3):
        for loop in range(1,nb+1):
            outData.GetRasterBand(loop).WriteArray(image[loop-1,:,:])
    if(image.ndim==2):
        outData.GetRasterBand(1).WriteArray(image)
    outData.FlushCache()