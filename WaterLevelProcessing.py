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
import scipy as sci

# %% Input - to be filled in individually
# your personal working directory in which you use the standard folder structure:
pwd = r'C:\Users\7062052\OneDrive - Universiteit Utrecht\Open Science Metronome\MetronomeWorkingDirectory'
# the current experiment (three digit number, in case of pilots use e.g. '052\Pilot1' as this string is used for directory paths)
exp = '049'
# the tilting period in seconds
tiltperiod = 40
# the samplng frequency of the water level sensors in Hertz
frequency = 10

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

#%% Filter for outliers and average over tidal cycles

tiltvalrange = np.percentile(alltiltdata["Range[mm]"],[2, 98]) #2nd and 98th percentile are extracted from all the tilting data
#the valid range for the tilting measurements is set to the 2nd and 98th percentile minus respectively plus 1/100 of the difference between them
tiltvalrange = [tiltvalrange[0]-(tiltvalrange[1]-tiltvalrange[0])/100, tiltvalrange[1]+(tiltvalrange[1]-tiltvalrange[0])/100]

wlvalrange = np.percentile(allwldata["Range[mm]"],[2, 98]) #2nd and 98th percentile are extracted from all the water level data
#the valid range for the water level measurements is set to the 2nd and 98th percentile minus respectively plus 1/100 of the difference between them
wlvalrange = [wlvalrange[0]-(wlvalrange[1]-wlvalrange[0])/100, wlvalrange[1]+(wlvalrange[1]-wlvalrange[0])/100]

tiltrange = [[] for j in range(len(tilting))] #create empty list for the averaged range measuremnts of the tilting data
bluerange = [[] for j in range(len(tilting))] #create empty list for the averaged range measuremnts of the water level data of the blue sensor
orangerange = [[] for j in range(len(tilting))] #create empty list for the averaged range measuremnts of the water level data of the orange sensor
greenrange = [[] for j in range(len(tilting))] #create empty list for the averaged range measuremnts of the water level data of the green sensor

#all values outside the valid range are set to nan in all the data
#loop through all the tilt files  
for i,it in enumerate(tiltdata):
    for j,jt in enumerate(tiltdata[i]):
        tiltdata[i][j] = jt.where(jt["Range[mm]"] > tiltvalrange[0]) #set all values outside the given condition nan
        tiltdata[i][j] = tiltdata[i][j].where(tiltdata[i][j]["Range[mm]"] < tiltvalrange[1]) #set all values outside the given condition nan
        tiltdata[i][j] = tiltdata[i][j].where(tiltdata[i][j]["Range[mm]"] != 0) #set all values outside the given condition nan
        fullcyclearray=[] #create empty list where the data of all fully measured tidal cycles can be stored
        for k in range(math.floor(len(tiltdata[i][j])/(frequency*tiltperiod))): #loop through the full cycles
            fullcyclearray.append(tiltdata[i][j][k*(frequency*tiltperiod):(k+1)*(frequency*tiltperiod)]["Range[mm]"].values) #append the measured values to the array
            
        tiltrange[i].append(np.nanmean(fullcyclearray, axis = 0)) #average the data of all full cycles and omit nan values
        #the last cycle is not covered completely, so only the corresponding part of the data of the full cycles is taken into account in taking the average
        tiltrange[i][j][0:(len(tiltdata[i][j])-(k+1)*frequency*tiltperiod)] = \
            np.nanmean(np.vstack((np.tile([tiltrange[i][j][0:len(tiltdata[i][j])-(k+1)*frequency*tiltperiod]],(k+1,1)), tiltdata[i][j][(k+1)*(frequency*tiltperiod):]["Range[mm]"].values)), axis = 0)
            
#loop through all the files of the blue sensor
for i,it in enumerate(bluedata):
    for j,jt in enumerate(bluedata[i]):
        bluedata[i][j] = jt.where(jt["Range[mm]"] > wlvalrange[0]) #set all values outside the given condition nan
        bluedata[i][j] = bluedata[i][j].where(bluedata[i][j]["Range[mm]"] < wlvalrange[1]) #set all values outside the given condition nan
        fullcyclearray=[] #create empty list where the data of all fully measured tidal cycles can be stored
        for k in range(math.floor(len(bluedata[i][j])/(frequency*tiltperiod))): #loop through the full cycles
            fullcyclearray.append(bluedata[i][j][k*(frequency*tiltperiod):(k+1)*(frequency*tiltperiod)]["Range[mm]"].values) #append the measured values to the array
            
        bluerange[i].append(np.nanmean(fullcyclearray, axis = 0)) #average the data of all full cycles and omit nan values
        #the last cycle is not covered completely, so only the corresponding part of the data of the full cycles is taken into account in taking the average
        bluerange[i][j][0:(len(bluedata[i][j])-(k+1)*frequency*tiltperiod)] = \
            np.nanmean(np.vstack((np.tile([bluerange[i][j][0:len(bluedata[i][j])-(k+1)*frequency*tiltperiod]],(k+1,1)), bluedata[i][j][(k+1)*(frequency*tiltperiod):]["Range[mm]"].values)), axis = 0)
        
#loop through all the files of the orange sensor        
for i,it in enumerate(orangedata):
    for j,jt in enumerate(orangedata[i]):
        orangedata[i][j] = jt.where(jt["Range[mm]"] > wlvalrange[0]) #set all values outside the given condition nan
        orangedata[i][j] = orangedata[i][j].where(orangedata[i][j]["Range[mm]"] < wlvalrange[1]) #set all values outside the given condition nan
        fullcyclearray=[] #create empty list where the data of all fully measured tidal cycles can be stored
        for k in range(math.floor(len(orangedata[i][j])/(frequency*tiltperiod))): #loop through the full cycles
            fullcyclearray.append(orangedata[i][j][k*(frequency*tiltperiod):(k+1)*(frequency*tiltperiod)]["Range[mm]"].values) #append the measured values to the array
            
        orangerange[i].append(np.nanmean(fullcyclearray, axis = 0)) #average the data of all full cycles and omit nan values
        #the last cycle is not covered completely, so only the corresponding part of the data of the full cycles is taken into account in taking the average
        orangerange[i][j][0:(len(orangedata[i][j])-(k+1)*frequency*tiltperiod)] = \
            np.nanmean(np.vstack((np.tile([orangerange[i][j][0:len(orangedata[i][j])-(k+1)*frequency*tiltperiod]],(k+1,1)), orangedata[i][j][(k+1)*(frequency*tiltperiod):]["Range[mm]"].values)), axis = 0)
                    
#loop through all the files of the green sensor
for i,it in enumerate(greendata):
    for j,jt in enumerate(greendata[i]):
        greendata[i][j] = jt.where(jt["Range[mm]"] > wlvalrange[0]) #set all values outside the given condition nan
        greendata[i][j] = greendata[i][j].where(greendata[i][j]["Range[mm]"] < wlvalrange[1]) #set all values outside the given condition nan
        fullcyclearray=[] #create empty list where the data of all fully measured tidal cycles can be stored
        for k in range(math.floor(len(greendata[i][j])/(frequency*tiltperiod))): #loop through the full cycles
            fullcyclearray.append(greendata[i][j][k*(frequency*tiltperiod):(k+1)*(frequency*tiltperiod)]["Range[mm]"].values) #append the measured values to the array
            
        greenrange[i].append(np.nanmean(fullcyclearray, axis = 0)) #average the data of all full cycles and omit nan values
        #the last cycle is not covered completely, so only the corresponding part of the data of the full cycles is taken into account in taking the average
        greenrange[i][j][0:(len(greendata[i][j])-(k+1)*frequency*tiltperiod)] = \
            np.nanmean(np.vstack((np.tile([greenrange[i][j][0:len(greendata[i][j])-(k+1)*frequency*tiltperiod]],(k+1,1)), greendata[i][j][(k+1)*(frequency*tiltperiod):]["Range[mm]"].values)), axis = 0)

#%% Smooth data
window = 100 #the filtering window for the Savitzky-Golay filter
polyord = 3 #polynomial's order for the Savitzky-Golay filter
funcx = lambda a : a.nonzero()[0] #functionion needed to interpolate nans

#loop through all the tilt files  
for i,it in enumerate(tiltrange):
    for j,jt in enumerate(tiltrange[i]):
        nans = np.isnan(tiltrange[i][j])
        tiltrange[i][j][nans]= np.interp(funcx(nans), funcx(~nans), tiltrange[i][j][~nans])
        tiltrange[i][j] = sci.signal.savgol_filter(tiltrange[i][j],window,polyord)
      
for i,it in enumerate(bluerange):
    for j,jt in enumerate(bluerange[i]):
        nans = np.isnan(bluerange[i][j])
        bluerange[i][j][nans]= np.interp(funcx(nans), funcx(~nans), bluerange[i][j][~nans])
        bluerange[i][j] = sci.signal.savgol_filter(bluerange[i][j],window,polyord)

for i,it in enumerate(orangerange):
    for j,jt in enumerate(orangerange[i]):
        nans = np.isnan(orangerange[i][j])
        orangerange[i][j][nans]= np.interp(funcx(nans), funcx(~nans), orangerange[i][j][~nans])
        orangerange[i][j] = sci.signal.savgol_filter(orangerange[i][j],window,polyord)

for i,it in enumerate(greenrange):
    for j,jt in enumerate(greenrange[i]):
        nans = np.isnan(greenrange[i][j])
        greenrange[i][j][nans]= np.interp(funcx(nans), funcx(~nans), greenrange[i][j][~nans])
        greenrange[i][j] = sci.signal.savgol_filter(greenrange[i][j],window,polyord)

#%% Sort data to normalize it to tilt
for i,it in enumerate(tiltrange):
    for j,jt in enumerate(tiltrange[i]):
        index = np.argmin(tiltrange[i][j]) #find index of peak flood
        end = tiltrange[i][j][0:index] #everything before peak flood
        start = tiltrange[i][j][index:] #everything after peak flood
        tiltrange[i][j] = np.hstack((start, end)) #rearrange with peak flood at the beginning
        #do the same with the water level data based on the peak flood index from the tilt
        end = bluerange[i][j][0:index]
        start = bluerange[i][j][index:]
        bluerange[i][j] = np.hstack((start, end))
        end = orangerange[i][j][0:index]
        start = orangerange[i][j][index:]
        orangerange[i][j] = np.hstack((start, end))
        end = greenrange[i][j][0:index]
        start = greenrange[i][j][index:]
        greenrange[i][j] = np.hstack((start, end))

#%% Putting all data together
#create array of zeros with the following dimensions: 
    # - x: number of measurements from the last measuring cycle, assuming that this will be the maximum number
    # - y: 3, the number of sensors
    # - timeseries: number of measuring points in one tidal cycle
    # - experiment time: number of measuring events throughout the experiment
wldata = np.zeros((len(bluerange[-1]), 3, frequency*tiltperiod, len(tiltingcycles)))

#Saving all water level data in one array:
for i in range(np.shape(wldata)[3]):
    for j in range(len(bluerange[i])):
        wldata[-j-1,0,:,i] = bluerange[i][-j-1]
        wldata[-j-1,1,:,i] = orangerange[i][-j-1]
        wldata[-j-1,2,:,i] = greenrange[i][-j-1]
        
#%% Saving data


#%% Plotting
time = np.linspace(0,tiltperiod,frequency*tiltperiod)
plt.plot(time,tiltrange[1][2])






                    
