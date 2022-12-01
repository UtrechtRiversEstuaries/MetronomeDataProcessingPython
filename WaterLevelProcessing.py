# -*- coding: utf-8 -*-
"""
This script processes the water level measurements from the Metronome.

The raw water level data is read from the corresponding directory and then:
 - filtered for outliers
 - averaged over the usually three measured cycles
 - sorted to align all data equally with the tilt
 - smoothed
 - put in relation to the still water level
 - saved as netCDF-file
 - plotted

Note: 
 - The data should be saved in subfolders thst are named e.g. 
     '1500cycles_tilting' and there should be at least one measurement of the 
     still water level
 - The final data does not contain the y-position of the sensors for the 
     different measurements. Make sure to have this data elsewhere.
 - The folder 'water_level' should already exist in the directory 
     'processed_data' of the corresponding experiment, otherwise the processed
     data is not saved.
 - The raw data files should be named as follows: 
     <cyclesnumber>_<x-metres>.<x-decimetres>_<tilting or still>_<automatic time stamp>_LT.s<sensornumber>.csv
    

Code written in November 2022

@author: Jan-Eike Rossius
"""

# %% Importing libraries
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import math  
import os
import pandas as pd
import scipy as sci
from datetime import datetime
import itertools

# %% Input - to be filled in individually
# your personal working directory in which you use the standard folder structure:
pwd = r'C:\Users\7062052\OneDrive - Universiteit Utrecht\Open Science Metronome\MetronomeWorkingDirectory'
# the current experiment (three digit number, in case of pilots use e.g. '052\Pilot1' as this string is used for directory paths)
exp = '051'
# the tilting period in seconds
tiltperiod = 40
# the samplng frequency of the water level sensors in Hertz
frequency = 10

# %% Read directory and files
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
xpos = [[] for j in range(len(tilting))] #create empty list for the x-positions of the measurements
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

files = os.listdir(wlpath + r'\\' + still[0]) #find content of the first folder with still water measurements. If there are more folders for still water measurements, they are ignored
#loop through files and sort them in arrays corresponding to sensor
for f in files:
    if '.s02' in f:
        greenstill = f
    elif '.s03' in f:
        orangestill = f
    elif '.s04' in f:
        bluestill = f
    
#loop through all the tilt files  
for i,it in enumerate(tiltfiles):
    for j,jt in enumerate(tiltfiles[i]):
        tempwldata = pd.read_csv(wlpath + r'\\' + tilting[i] + r'\\' + jt) #read the data
        tiltdata[i].append(tempwldata) #store the data in a similar structure as the file names
        if 'alltiltdata' in locals(): #if alltiltdata already exists:
            alltiltdata = pd.concat([alltiltdata,tempwldata]) #append the data of the current file to all the previos data
        else: #if alltiltdata does not yet exist, it is created
            alltiltdata = tempwldata
        try:
            xpos[i].append(float(jt[jt.find(".")-2:jt.find(".")+2]))
        except ValueError:
            xmetre = int(jt[jt.find("-")-2:jt.find("-")])
            xdecimetre = int(jt[jt.find("-")+1:jt.find("-")+2])
            xpos[i].append((xmetre + xdecimetre / 10))

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

#read data of the still water measurement
bluestill = pd.read_csv(wlpath + r'\\' + still[0] + r'\\' + bluestill)
orangestill = pd.read_csv(wlpath + r'\\' + still[0] + r'\\' + greenstill)
greenstill = pd.read_csv(wlpath + r'\\' + still[0] + r'\\' + greenstill)

# %% Filter for outliers and average over tidal cycles

tiltvalrange = np.percentile(alltiltdata["Range[mm]"],[2, 98]) #2nd and 98th percentile are extracted from all the tilting data
#the valid range for the tilting measurements is set to the 2nd and 98th percentile minus respectively plus 1/100 of the difference between them
tiltvalrange = [tiltvalrange[0]-(tiltvalrange[1]-tiltvalrange[0])/100, tiltvalrange[1]+(tiltvalrange[1]-tiltvalrange[0])/100]

wlvalrange = np.percentile(allwldata["Range[mm]"],[10, 90]) #2nd and 98th percentile are extracted from all the water level data
#the valid range for the water level measurements is set to the 2nd and 98th percentile minus respectively plus 1/100 of the difference between them
wlvalrange = [wlvalrange[0]-(wlvalrange[1]-wlvalrange[0])/10, wlvalrange[1]+(wlvalrange[1]-wlvalrange[0])/10]

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

#still water data
bluestill = bluestill.where(bluestill["Range[mm]"] > wlvalrange[0]) #set all values outside the given condition nan
bluestill = bluestill.where(bluestill["Range[mm]"] < wlvalrange[1]) #set all values outside the given condition nan
orangestill = orangestill.where(orangestill["Range[mm]"] > wlvalrange[0]) #set all values outside the given condition nan
orangestill = orangestill.where(orangestill["Range[mm]"] < wlvalrange[1]) #set all values outside the given condition nan
greenstill = greenstill.where(greenstill["Range[mm]"] > wlvalrange[0]) #set all values outside the given condition nan
greenstill = greenstill.where(greenstill["Range[mm]"] < wlvalrange[1]) #set all values outside the given condition nan
#average data
stillwaterlevel = np.nanmean(np.hstack((bluestill["Range[mm]"].values, orangestill["Range[mm]"].values, greenstill["Range[mm]"].values)))

# %% Sort data to normalize it to tilt
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
        
# %% Smooth data
window = 50 #the filtering window for the Savitzky-Golay filter
polyord = 3 #polynomial's order for the Savitzky-Golay filter
funcx = lambda a : a.nonzero()[0] #functionion needed to interpolate nans

#loop through all the tilt files  
for i,it in enumerate(tiltrange):
    for j,jt in enumerate(tiltrange[i]):
        nans = np.isnan(tiltrange[i][j])
        tiltrange[i][j][nans]= np.interp(funcx(nans), funcx(~nans), tiltrange[i][j][~nans]) #interpolate nan values
        tiltrange[i][j] = sci.signal.savgol_filter(tiltrange[i][j],window,polyord) #smooth the data with a Savitzki-Golay-filter
      
for i,it in enumerate(bluerange):
    for j,jt in enumerate(bluerange[i]):
        try:
            nans = np.isnan(bluerange[i][j])
            bluerange[i][j][nans]= np.interp(funcx(nans), funcx(~nans), bluerange[i][j][~nans]) #interpolate nan values
            bluerange[i][j] = sci.signal.savgol_filter(bluerange[i][j],window,polyord) #smooth the data with a Savitzki-Golay-filter
        except ValueError:
            print('A measurement with only invalid values occured')

for i,it in enumerate(orangerange):
    for j,jt in enumerate(orangerange[i]):
        try:
            nans = np.isnan(orangerange[i][j])
            orangerange[i][j][nans]= np.interp(funcx(nans), funcx(~nans), orangerange[i][j][~nans]) #interpolate nan values
            orangerange[i][j] = sci.signal.savgol_filter(orangerange[i][j],window,polyord) #smooth the data with a Savitzki-Golay-filter
        except ValueError:
            print('A measurement with only invalid values occured')

for i,it in enumerate(greenrange):
    for j,jt in enumerate(greenrange[i]):
        try:
            nans = np.isnan(greenrange[i][j])
            greenrange[i][j][nans]= np.interp(funcx(nans), funcx(~nans), greenrange[i][j][~nans]) #interpolate nan values
            greenrange[i][j] = sci.signal.savgol_filter(greenrange[i][j],window,polyord) #smooth the data with a Savitzki-Golay-filter
        except ValueError:
            print('A measurement with only invalid values occured')
            
# %% Substracting values from still waterlevel to have actual water levels and not measured ranges
for i,it in enumerate(bluerange):
    for j,jt in enumerate(bluerange[i]):
        bluerange[i][j] = stillwaterlevel - bluerange[i][j]

for i,it in enumerate(orangerange):
    for j,jt in enumerate(orangerange[i]):
        orangerange[i][j] = stillwaterlevel - orangerange[i][j]
        
for i,it in enumerate(greenrange):
    for j,jt in enumerate(greenrange[i]):
        greenrange[i][j] = stillwaterlevel - greenrange[i][j]
        
# %% Putting all data together
#find unique x-positions
xunique = np.unique(np.array(list(itertools.chain(*xpos))))

#create array of zeros with the following dimensions: 
    # - x: number of measurements from the last measuring cycle, assuming that this will be the maximum number
    # - y: 3, the number of sensors
    # - timeseries: number of measuring points in one tidal cycle
    # - experiment time: number of measuring events throughout the experiment
wldata = np.full((len(xunique), 3, frequency*tiltperiod, len(tiltingcycles)),np.nan)

#Saving all water level data in one array:
for i in range(np.shape(wldata)[3]):
    for j in range(len(bluerange[i])):
        wlx = np.where(xunique==xpos[i][j])[0][0] #find correct x-position
        wldata[wlx,0,:,i] = bluerange[i][j]
        wldata[wlx,1,:,i] = orangerange[i][j]
        wldata[wlx,2,:,i] = greenrange[i][j]

#creating a variable for the time (measured time series)
time = np.linspace(0,tiltperiod,frequency*tiltperiod)

#creating a variable for the tilt
for i,it in enumerate(tiltrange):
    for j,jt in enumerate(tiltrange[i]):
        if i==0 & j==0:
            alltilt = tiltrange[i][j]
        else:
            alltilt=np.vstack([alltilt, tiltrange[i][j]])
            
tilt = np.nanmean(alltilt, axis = 0) #mean of all tilt curves as they are all aligned
        
# %% Saving data
#creating netCDF-file for storing the data
ncfile = nc.Dataset(pwd + r'\01metronome_experiments\Exp' + exp + r'\processed_data\water_level\WaterLevelDataExp' + exp + '.nc', 'w', format='NETCDF4')

#adding global attributes
ncfile.Conventions = 'mostly CF-1.8'
ncfile.title = 'Water level data of Metronome experiment ' + exp
ncfile.institution = 'Utrecht University'
ncfile.source = 'Metronome ultrasonic water level sensors'
ncfile.history = 'Processing of the raw data by filtering outliers, averaging over tidal cycles, smoothing and aligning to the tilt in Python on: ' + str(datetime.utcnow())
ncfile.references = ''
ncfile.comment = ''

#create dimensios
ncfile.createDimension('experiment_time', None)
ncfile.createDimension('x', len(xunique))
ncfile.createDimension('sensors', 3)
ncfile.createDimension('measuring_time', frequency*tiltperiod)

#create dimension variables
x_var = ncfile.createVariable('x', np.float32, ('x',))
x_var[:] = xunique
x_var.units = 'm'
x_var.axis = 'X' 
x_var.long_name = 'x-locations along the flume of the measurements'

sens_var = ncfile.createVariable('sensors', str, ('sensors',))
sens_var[:] = np.array(['blue','orange','green'])
sens_var.units = '-'
sens_var.axis = 'Y' 
sens_var.long_name = 'identifyer for the three individual water level sensors'

etime_var = ncfile.createVariable('experiment_time', np.float32, ('experiment_time',))
etime_var[:] = tiltingcycles
etime_var.units = 'tidal_cycles'
etime_var.axis = 'T' 
etime_var.long_name = 'time throughout the course of the experiment'

mtime_var = ncfile.createVariable('measuring_time', np.float32, ('measuring_time',))
mtime_var[:] = time
mtime_var.units = 'seconds'
mtime_var.long_name = 'time of the measured time series'

#creating tilt variable
tilt_var = ncfile.createVariable('tilt_amplitude', np.float32, ('measuring_time',))
tilt_var[:] = tilt
tilt_var.units = 'mm'
tilt_var.long_name = 'amplitude of the Metronome flume tilt at x=0'

#creating water level variable
wl_var = ncfile.createVariable('water_level', np.float32, ('x','sensors','measuring_time','experiment_time'))
wl_var[:] = wldata
wl_var.units = 'mm'
wl_var.long_name = 'measured range of the water level sensors'

#closing the netCDF-file
ncfile.close()

# %% Plotting
# Defining general variables for plotting
figheight = 450
figwidth = 1200
dpi = 150
ymin = stillwaterlevel - wlvalrange[1] - 3 #upper limit of the water level plots
ymax = stillwaterlevel - wlvalrange[0] + 3 #lower limit of the water level plots
gridsize = (3, len(xunique)+2) #the gridsize for the subplots

for step,cycle in enumerate(tiltingcycles):
    fig = plt.figure(figsize=(figwidth/dpi, figheight/dpi), dpi=dpi) #creating the figure
    ax = [None] * len(xunique) * 3 # pre-allocating array for axes objects
    
    #plotting the tilting curve
    axtilt = plt.subplot2grid(gridsize, (1, 0))
    axtilt.set(xticks=[0, tiltperiod/2, tiltperiod], yticks=[round(tiltvalrange[0],-1), 0, round(tiltvalrange[1],-1)], xlabel='time since\nstart of\ntidal cycle [s]', ylabel='tilting amplitude\nat x=0 [mm]')
    axtilt.plot(time,tilt, color='black')
    axtilt.set_xlim(0,tiltperiod)
    axtilt.set_ylim(min([tiltvalrange[0],round(tiltvalrange[0],-1)]), max([tiltvalrange[1],round(tiltvalrange[1],-1)]))
    axtilt.text(tiltperiod/2,tiltvalrange[0]+25,'flood',verticalalignment='center',horizontalalignment='center')
    axtilt.text(tiltperiod/2,tiltvalrange[1]-25,'ebb',verticalalignment='center',horizontalalignment='center')
    axtilt.text(-tiltperiod,3*tiltvalrange[1],str(cycle) + ' cycles',fontsize=14)
    
    for i in range(np.shape(wldata)[0]): #loop through the x-positions
        ax[i] = plt.subplot2grid(gridsize, (0, i+2)) #create upper plot
        ax[i].set_title(str(xunique[i]) + ' m') #x-position as title
        ax[i].set(xticks=[])
        if i>0: #no yticks except for lefternmost plot
            ax[i].set(yticks=[])
        ax[len(xunique)+i] = plt.subplot2grid(gridsize, (1, i+2)) #create middle plot
        if i==0: #lefternmost plot gets yticks and ylabel
            ax[len(xunique)+i].set(xticks=[])
            ax[len(xunique)+i].set(ylabel='water level with respect to the\nstill water level [mm]')
        else:
            ax[len(xunique)+i].set(xticks=[], yticks=[])
        ax[2*len(xunique)+i] = plt.subplot2grid(gridsize, (2, i+2)) #create lower plot
        if i>0: #no yticks except for lefternmost plot
            ax[2*len(xunique)+i].set(yticks=[])
        if i==math.floor(len(xunique)/2): #xlabel roughly in the middle
            ax[2*len(xunique)+i].set(xlabel='time since start of tidal cycle [s]')
        ax[2*len(xunique)+i].set(xticks=[tiltperiod/4, 3*tiltperiod/4]) #define xticks
        ax[i].plot(time,wldata[i,2,:,step], color='green') #set colour
        ax[len(xunique)+i].plot(time,wldata[i,1,:,step], color='orange') #set colour
        ax[2*len(xunique)+i].plot(time,wldata[i,0,:,step], color='blue') #set colour
        ax[i].set_xlim(0,tiltperiod) #set xlimits
        ax[len(xunique)+i].set_xlim(0,tiltperiod) #set xlimits
        ax[2*len(xunique)+i].set_xlim(0,tiltperiod) #set xlimits
        ax[i].set_ylim(ymin,ymax) #set ylimits
        ax[len(xunique)+i].set_ylim(ymin,ymax) #set ylimits
        ax[2*len(xunique)+i].set_ylim(ymin,ymax) #set ylimits
