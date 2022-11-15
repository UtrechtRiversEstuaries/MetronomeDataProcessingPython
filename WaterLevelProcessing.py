# -*- coding: utf-8 -*-
"""
This script processes the water level measurements from the Metronome.
Created on Wed Nov  9 11:49:51 2022

@author: Jan-Eike Rossius
"""

# %% importing libraries
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from IPython import get_ipython; 
import math  
import ctypes
import statistics
import os
import pandas as pd

# %% Input - to be filled in individually
# your personal working directory in which you use the standard folder structure:
pwd = r'C:\Users\7062052\OneDrive - Universiteit Utrecht\Open Science Metronome\MetronomeWorkingDirectory'
# the current experiment (three digit number, in case of pilots use e.g. '052\Pilot1' as this string is used for directory paths)
exp = '049'

# %% read directory and files
wlpath = pwd + r'\01metronome_experiments\Exp' + exp + r'\raw_data\waterlevel_sensors' #path to the water level directory
elems = os.listdir(wlpath) #read elements in directory
tilting = [] #create empty list for tilting measuremens directory
still = [] #create empty list for still measurements directory
tiltingcycles = [] #create empty list for tilting measurements cyclenumbers
stillcycles = [] #create empty list for still measurements cyclenumbers

for e in elems: #loop through elements of directory
    if os.path.isfile(os.path.join(wlpath, e))==False: #only evaluate folders
        if 'tilting' in e: #if folder name contains tilting
            tilting.append(e) #store folder name
            for idx,let in enumerate(e): #check how many chars are numbers
                if let.isnumeric():
                    endnum = idx+1
                else:
                    break
            numstring = e[0:endnum]
            tiltingcycles.append(int(numstring)) #add the cycle number to the tilting list
        elif 'still' in e: #if folder name contains still
            still.append(e) #store folder name
            for idx,let in enumerate(e): #check how many chars are numbers
                if let.isnumeric():
                    endnum = idx+1
                else:
                    break
            numstring = e[0:endnum]
            stillcycles.append(int(numstring)) #add the cycle number to the still list

nummes = [] #create empty list for number of measuremnts along the flume per cycle
tiltfiles = [[] for j in range(len(tilting))] #create empty list for files containing the tilting data
bluefiles = [[] for j in range(len(tilting))] #create empty list for files containing the water level data of the blue sensor (hall side, smaller y)
orangefiles = [[] for j in range(len(tilting))] #create empty list for files containing the water level data of the orange sensor (middle, medium y)
greenfiles = [[] for j in range(len(tilting))] #create empty list for files containing the water level data of the green sensor (wall side, higher y)

tiltdata = [[] for j in range(len(tilting))] #create empty list for the tilting data
bluedata = [[] for j in range(len(tilting))] #create empty list for the water level data of the blue sensor (hall side, smaller y)
orangedata = [[] for j in range(len(tilting))] #create empty list for the water level data of the orange sensor (middle, medium y)
greendata = [[] for j in range(len(tilting))] #create empty list for the water level data of the green sensor (wall side, higher y)

for i,t in enumerate(tilting): #loop thtough all measurement timesteps
    files = os.listdir(wlpath + r'\\' + t) #list files
    nummes.append(len(files)/4) #number of files divided by 4 gives number of measurement locations along the flume
      
    #loop through files and sort them in arrays corresponding to sensor
    for f in files:
        if '.s00' in f:
            tiltfiles[i].append(f)
        elif '.s02' in f:
            greenfiles[i].append(f)
        elif '.s03' in f:
            orangefiles[i].append(f)
        elif '.s04' in f:
            bluefiles[i].append(f)
   
#loop through all the tilt files  
for i,it in enumerate(tiltfiles):
    for j,jt in enumerate(tiltfiles[i]):
        tempwldata = pd.read_csv(wlpath + r'\\' + tilting[i] + r'\\' + jt) #read the data
        tiltdata[i].append(tempwldata) #store the data in a similar structure as the file names
        if 'alltiltdata' in locals(): #if alltiltdata already exists:
            alltiltdata = pd.concat([alltiltdata,tempwldata]) #append the data of the current file to all the previos data
        else: #if alltiltdata does not yet exist, it is created
            alltiltdata = tempwldata

#loop through all the files of the blue sensor
for i,it in enumerate(bluefiles):
    for j,jt in enumerate(bluefiles[i]):
        tempwldata = pd.read_csv(wlpath + r'\\' + tilting[i] + r'\\' + jt) #read the data
        bluedata[i].append(tempwldata) #store the data in a similar structure as the file names
        if 'allwldata' in locals(): #if allwldata already exists:
            allwldata = pd.concat([allwldata,tempwldata]) #append the data of the current file to all the previos data
        else: #if allwldata does not yet exist, it is created
            allwldata = tempwldata

#loop through all the files of the orange sensor        
for i,it in enumerate(orangefiles):
    for j,jt in enumerate(orangefiles[i]):
        tempwldata = pd.read_csv(wlpath + r'\\' + tilting[i] + r'\\' + jt) #read the data
        orangedata[i].append(tempwldata) #store the data in a similar structure as the file names
        allwldata = pd.concat([allwldata,tempwldata]) #append the data of the current file to all the previos data

#loop through all the files of the green sensor
for i,it in enumerate(greenfiles):
    for j,jt in enumerate(greenfiles[i]):
        tempwldata = pd.read_csv(wlpath + r'\\' + tilting[i] + r'\\' + jt) #read the data
        greendata[i].append(tempwldata) #store the data in a similar structure as the file names
        allwldata = pd.concat([allwldata,tempwldata]) #append the data of the current file to all the previos data

#%% Filter for outliers

tiltvalrange = np.percentile(alltiltdata["Range[mm]"],[2, 98]) #2nd and 98th percentile are extracted from all the tilting data
#the valid range for the tilting measurements is set to the 2nd and 98th percentile minus respectively plus 1/100 of the difference between them
tiltvalrange = [tiltvalrange[0]-(tiltvalrange[1]-tiltvalrange[0])/100, tiltvalrange[1]+(tiltvalrange[1]-tiltvalrange[0])/100]

wlvalrange = np.percentile(allwldata["Range[mm]"],[2, 98]) #2nd and 98th percentile are extracted from all the water level data
#the valid range for the water level measurements is set to the 2nd and 98th percentile minus respectively plus 1/100 of the difference between them
wlvalrange = [wlvalrange[0]-(wlvalrange[1]-wlvalrange[0])/100, wlvalrange[1]+(wlvalrange[1]-wlvalrange[0])/100]

#all values outside the valid range are set to nan in all the data
#loop through all the tilt files  
for i,it in enumerate(tiltdata):
    for j,jt in enumerate(tiltdata[i]):
        tiltdata[i][j] = jt.where(jt["Range[mm]"] > tiltvalrange[0]) #set all values outside the given condition nan
        tiltdata[i][j] = tiltdata[i][j].where(tiltdata[i][j]["Range[mm]"] < tiltvalrange[1]) #set all values outside the given condition nan
        tiltdata[i][j] = tiltdata[i][j].where(tiltdata[i][j]["Range[mm]"] != 0) #set all values outside the given condition nan
            
#loop through all the files of the blue sensor
for i,it in enumerate(bluedata):
    for j,jt in enumerate(bluedata[i]):
        bluedata[i][j] = jt.where(jt["Range[mm]"] > wlvalrange[0]) #set all values outside the given condition nan
        bluedata[i][j] = bluedata[i][j].where(bluedata[i][j]["Range[mm]"] < wlvalrange[1]) #set all values outside the given condition nan
        
#loop through all the files of the orange sensor        
for i,it in enumerate(orangedata):
    for j,jt in enumerate(orangedata[i]):
        orangedata[i][j] = jt.where(jt["Range[mm]"] > wlvalrange[0]) #set all values outside the given condition nan
        orangedata[i][j] = orangedata[i][j].where(orangedata[i][j]["Range[mm]"] < wlvalrange[1]) #set all values outside the given condition nan
                    
#loop through all the files of the green sensor
for i,it in enumerate(greendata):
    for j,jt in enumerate(greendata[i]):
        greendata[i][j] = jt.where(jt["Range[mm]"] > wlvalrange[0]) #set all values outside the given condition nan
        greendata[i][j] = greendata[i][j].where(greendata[i][j]["Range[mm]"] < wlvalrange[1]) #set all values outside the given condition nan
                    
