'''
Demo code to load interval Water Depth Maps from the dataset of experimental estuaries (Nota et al., 2026)

Note that all experimental metadata is loaded from a separate file. 
Accordingly, only the indices of the experiments of interest need to be set in this code (line 42)

Author: Eise W. Nota (finalized MAY 2026)   

'''

#%% Clearing all variables
from IPython import get_ipython
get_ipython().magic('reset -sf') 

#%% Import packages
import os
import pandas as pd
import datetime
import netCDF4 as nc
import numpy as np
from datetime import timedelta
# For optional plotting:
import matplotlib.pyplot as plt
import matplotlib
from pandas import Timedelta


#%% Set base directory of the dataset network drive (here set to align with the folder structure in the repository)
pwd = r"\\...\original"

# Set the directory of the functions
start = datetime.datetime.now()

#%% Define fixed variables 
baseModel = '11'        # Base Model within which geometry the data was processed   
gridRes = '25'          # Desired grid resolution in mm; fixed at 25 for water depth maps

#%% Read metadata
metaData = pd.read_excel(pwd + r"\\expMetaData.xlsx") 

# Which experiments (indices) do we want to compute?
expsi = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20] # Min 0; Max 20

# Make list of experimental directories
expFolders = os.listdir(pwd + '\\metronome_experiments')

# Loop over the experiments of interest
for expi in expsi:
    print("Computing " + metaData.loc[expi,'Exp'])
    
    #%%Take the experiment-specific metadata
    Expnr = metaData.loc[expi,'Expnr'][:-1]                                                         # Experiment number
    pilot = metaData.loc[expi,'pilot']                                                              # Pilot number, 0 if no pilot
    start_interval = metaData.loc[expi,'start_interval'][:-1]                                       # String of orthomosaic cycle where you want to start # 00000 = first cycle
    end_interval = metaData.loc[expi,'end_interval'][:-1]                                           # String of orthomosaic cycle where you want to end # End with complete numbers as in folder structure 
    conc = metaData.loc[expi,'conc'][1:-1].replace("'", "").split(",")                              # Experimental dye concentration(s)
    concRange = metaData.loc[expi,'conc_range'][1:-1].replace("'", "").split(",")                   # Cycle ranges in case of multiple experimental dye concentrations                                             
    jump = metaData.loc[expi,'jump']                                                                # Interval of timesteps at timelapse
    cycleDurat = metaData.loc[expi,'cycleDurat']                                                    # Duration of one tidal cycle (s)
    maskExpnr = metaData.loc[expi,'maskExp'][:-1]                                                   # Mask experiment
    maskCycle = metaData.loc[expi,'maskCycle'][:-1]                                                 # Mask cycle
    maskBuffer = metaData.loc[expi,'maskBuffer'][:-1]                                               # Mask buffer
    maskpilot = metaData.loc[expi,'maskpilot']                                                      # Mask pilot
    cyclesToSkip = metaData.loc[expi,'cyclesToSkip'][1:-1].replace("'", "").split(",")              # Cycles not to be included
    cyclesRangesToSkip = metaData.loc[expi,'ignoreCycleRanges'][1:-1].replace("'", "").split(",")   # Already computed, but to be excluded
    cycles1Hz = metaData.loc[expi,'1Hz'][1:-1].replace("'", "").split(",")                          # Cycles of 1Hz overhead imagery
    mask1Hz = metaData.loc[expi,'mask1Hz'][1:-1].replace("'", "").split(",")                        # Optionally different masks for 1Hz overhead imagery
    cyclesPIV = metaData.loc[expi,'PIV'][1:-1].replace("'", "").split(",")                          # Cycles of 1Hz imagery
    
    # Skip iteration if there are no water depth maps
    if int(end_interval) - int(start_interval) == 0:
        continue
    
    # Define label start for relevant folders
    # Experimental folder
    if pilot == 0:
        expStart = 'Exp' + Expnr
    else:
        expStart = 'Exp' + Expnr + '_Pilot' + str(pilot)
    # Mask folder
    if maskpilot == 0:
        maskStart = 'Exp' + maskExpnr
    else:
        maskStart = 'Exp' + maskExpnr + '_Pilot' + str(maskpilot)
    # Define appropriate indices
    expIndex = [expFolders.index(l) for l in expFolders if l.startswith(expStart)][0]
    maskIndex = [expFolders.index(l) for l in expFolders if l.startswith(maskStart)][0]
    
    # Determine mask name
    if maskpilot == 0:
        if int(maskBuffer) > 0:
            maskName = 'MaskExp' + maskExpnr + 'Cycle' + maskCycle + 'Buffer' + maskBuffer
        else:
            maskName = 'MaskExp' + maskExpnr + 'Cycle' + maskCycle
    else:
        if int(maskBuffer) > 0:
            maskName = 'MaskExp' + maskExpnr + 'Pilot' + str(maskpilot) + 'Cycle' + maskCycle + 'Buffer' + maskBuffer
        else:
            maskName = 'MaskExp' + maskExpnr + 'Pilot' + str(maskpilot) + 'Cycle' + maskCycle
    
    #%% Define the relevant folders for mask and Water Depth Maps
    # Current experiment folder
    expFolder = pwd + '\\metronome_experiments\\' + expFolders[expIndex]
    maskFolder = pwd + '\\metronome_experiments\\' + expFolders[maskIndex]
    # Add subsequent folders
    expFolder = expFolder + '\\Overhead Water Depth Maps\\BaseModel' + baseModel + '\\Res' + gridRes + 'mm'
    maskFolder = maskFolder + '\\Masks\\BaseModel' + baseModel + '\\Res' + gridRes + 'mm'
    
    # Defining the depthMapFolders based on the amount of available dye concentrations
    depthMapFolders = []
    for c in conc:
        depthMapFolders.append(expFolder + '\\Conc' + c + '\\' + maskName.replace('Exp',''))
    
    #%% Load Mask
    maskDir = maskFolder + '\\' + maskName + '.nc'
    maskNC = nc.Dataset(maskDir, 'r')
    # Extract the relevant information
    mask = np.array(maskNC.get_variables_by_attributes(name='mask')[0])
    xAxis = np.array(maskNC.get_variables_by_attributes(name='X-axis')[0])
    yAxis = np.array(maskNC.get_variables_by_attributes(name='Y-axis')[0])
    # Convert the X/Y-axis information to plottable values
    xValues = np.arange(xAxis[0],xAxis[1],xAxis[2])
    yValues = np.arange(yAxis[0],yAxis[1],yAxis[2])
    xx, yy = np.meshgrid(xValues,yValues)
    # Flip yy
    yy = np.flipud(yy)
    # We can close the file again
    maskNC.close()
    # Flip vertically for ortho
    mask = np.flipud(mask)
    # Make mask suited for 3D
    orthoMask0 = np.zeros([np.size(mask,0),np.size(mask,1),3])
    orthoMask0[:,:,0] = mask; orthoMask0[:,:,1] = mask; orthoMask0[:,:,2] = mask
    # Also create a stacked mask
    maskStacked = mask.reshape(-1)
    boolMaskStacked = maskStacked.astype(bool)
    
    # Set zeros to nan
    nanmask = mask.astype(float)
    nanmask[nanmask==0] = np.nan
    # Create mesh with deleted columns with all nans
    xxnan = xx[:,~np.all(np.isnan(nanmask), axis=0)]
    yynan = yy[:,~np.all(np.isnan(nanmask), axis=0)]
    nanmask = nanmask[:,~np.all(np.isnan(nanmask), axis=0)]
    
    #####################################################
    # # Optionally visualize the mask
    # fig = plt.figure(figsize=(30,4.5))
    # ax = fig.gca()
    # ax.pcolormesh(xxnan, yynan, nanmask, rasterized=True)
    # ax.axes.set_aspect("equal")
    # plt.xlim(0,20)
    # plt.ylim(0,3)
    # plt.tight_layout()
    # plt.tick_params(labelsize=12.5)
    # plt.xlabel('x (m)', size=15)
    # plt.ylabel('y (m)', size=15)
    # plt.title(maskName, size=30)
    #####################################################
    
    #%% Define the relevant Depth Maps and their information
    # Update relevant dye concentrations for Exp063 (cycles ranges of 1Hz imagery and varying dye concentrations are excluded from the interval)
    if Expnr == '063' and pilot == 0:
        depthMapFolders = depthMapFolders[:1]
    
    # Determine the available files in the folder
    avail_files = []
    avail_fileDirs = []
    # Create list of available depth maps
    for depthMapFolder in depthMapFolders:
        avail_files.append(os.listdir(depthMapFolder))
        avail_fileDirs.append([os.path.join(depthMapFolder, file) for file in os.listdir(depthMapFolder)])
    # Stack avail_files to single list
    avail_files = sum(avail_files, [])   
    avail_fileDirs = sum(avail_fileDirs, [])   
    
    # Determine relevant depth maps
    rel_depthMaps = []
    rel_timesteps = []
    for i in range(len(avail_files)):
        file = avail_files[i]
        # First test whether the cycles occur in the cyclesToSkip list
        # Cycle number occurs at [-8:-3]
        if file[-8:-3] not in cyclesToSkip:
            if int(start_interval) <= int(file[-8:-3]) <= int(end_interval): 
                rel_depthMaps.append(avail_fileDirs[i])
                rel_timesteps.append(int(file[-8:-3]))

    rel_timesteps = np.array(rel_timesteps)
    rel_timestamps = rel_timesteps * cycleDurat # seconds
    # Sort
    rel_depthMaps = sorted(rel_depthMaps, key=lambda x: x.split('_')[2])
    rel_timesteps = sorted(rel_timesteps)
    rel_timestamps = sorted(rel_timestamps)

    # Do a final filtering on cyclesRangesToSkip, in case they are set
    if len(cyclesRangesToSkip[0]) != 0:
        # Reversed for loop to maintain proper indices, so "-1,-1,-1)"
        for i in range(len(rel_timesteps)-1,-1,-1):
            for currentRange in cyclesRangesToSkip:
                if int(currentRange[:5]) <= rel_timesteps[i] < int(currentRange[-5:]):
                    rel_timesteps.pop(i)
                    rel_depthMaps.pop(i)
                    rel_timestamps.pop(i)
        
    #%% Load all depth Maps into a single list
    depthMaps = []
    # Loop over the relevant depthMaps
    for i in range(len(rel_depthMaps)):
        currentDepthMap = rel_depthMaps[i]
        # Load currentDepthMap
        netCDFFile = nc.Dataset(currentDepthMap)
        # Extract the relevant information
        depthData = np.array(netCDFFile.get_variables_by_attributes(name='Z-axis')[0])
        # Reshape to align with plottable data
        depthData = depthData[:,:,0].reshape((depthData.shape[0],depthData.shape[1]))
        # We can close the file again
        netCDFFile.close()
        # Append
        depthMaps.append(depthData)
        
        # Make classifier map
        clfMap = -1*((nanmask*depthData)-1)
        # Set wet values to nan
        clfMap[clfMap<1] = np.nan
        
        #####################################################
        # # Optionally visualize the depthData
        # fig, ax = plt.subplots(figsize=(30,4.5),layout='constrained')
        # im = ax.pcolormesh(xxnan, yynan, nanmask*depthData*100, rasterized=True, vmin = 0, vmax = 7, cmap='Blues')  # ax.pcolormesh(xx, yy, nanmask, rasterized=True) 
        # im2 = ax.pcolormesh(xxnan, yynan, clfMap, cmap=matplotlib.colors.ListedColormap(['palegoldenrod']), rasterized=True)
        # ax.axes.set_aspect("equal")
        # ax.set_xlim(0,20)
        # ax.set_ylim(0,3)
        # ax.tick_params(labelsize=12.5)
        # ax.set_xlabel('x (m)', size=15)
        # ax.set_ylabel('y (m)', size=15)
        # ax.set_title(expStart + '   Timestep ' + "{0:05d}".format(rel_timesteps[i]) + '   (' + str(Timedelta(seconds=rel_timestamps[i])) + ')', size=30)
        # clb = fig.colorbar(im, ax=ax,location='right', pad = 0.005, shrink = 0.9)
        # clb.set_label('Water depth (cm)', size=20)
        # clb.ax.tick_params(labelsize=15)
        #####################################################
        
    #%% Now conduct the desired analysis on the Water Depth Maps
    '''
    From here on you can write your own code to conduct desired analysis using the following variables:
        
    - depthMaps : list of all depth maps within the experiment with water depths in m
    - rel_timesteps : associated timesteps (these equal tidal cycle numbers for Exp052-Exp078)
    - rel_timestamps : associated timestamps in seconds from the beginning of the experiment
    - nanmask : mask that has been applied to the water depth maps
    - xxnan, yynan : associated meshgrid
    - mask, xx, yy : above variables on the original coordinate system of the Metronome (from 0 < x < 20; and 0 < y < 3)
    
    '''
    
#%% How long did the run take?
end = datetime.datetime.now()
td = round((end - start).total_seconds())
print("Script running time = " + str(timedelta(seconds=td)))
