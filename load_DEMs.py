'''
Demo code to load DEMs from the dataset of experimental estuaries (Nota et al., 2026)

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

#%% Set base directory of the dataset network drive (here set to align with the folder structure in the repository)
pwd = r"\\...\original"

# Set the directory of the functions
start = datetime.datetime.now()

#%% Set desired resolutions
demRess = ['25','5']  # List of desired DEM resolutions in mm; options: '25'; '5'

#%% Define fixed variables
baseModel = '11'        # Base Model within which geometry the data was processed   
coords = [0,20,0,3]     # Metronome coordinate system [Xmin, Xmax, Ymin, Ymax]

#%% Read metadata
metaData = pd.read_excel(pwd + r"\\expMetaData.xlsx") 

# Which experiments (indices) do we want to compute?
expsi = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]   # Min 0; Max 20

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
    cyclesPIV = metaData.loc[expi,'PIV'][1:-1].replace("'", "").split(",")                          # Cycles of 1Hz imagery
    
    # Experimental folder
    if pilot == 0:
        expStart = 'Exp' + Expnr
    else:
        expStart = 'Exp' + Expnr + '_Pilot' + str(pilot)
    # Define appropriate indices
    expIndex = [expFolders.index(l) for l in expFolders if l.startswith(expStart)][0]
    
    #%% Defining the relevant folders for DEMs
    demFolders = []
    for demRes in demRess:
        demFolders.append(pwd + '\\metronome_experiments\\' + expFolders[expIndex] + '\\DEMs\\BaseModel' + baseModel + '\\Res' + demRes + 'mm')
        
    #%% Define the relevant DEMs and their information
    avail_files = []
    avail_fileDirs = []
    # Create list of available DEMs
    for demFolder in demFolders:
        avail_files.append(os.listdir(demFolder))
        avail_fileDirs.append([os.path.join(demFolder, file) for file in os.listdir(demFolder)])
    # All DEMs are relevant, so no filtering is required
    
    #%% Load all DEMs into a single list
    dems = []
    xxs = []
    yys = []
    # Loop over the relevant resolutions
    for j in range(len(avail_files)):
        demRes = demRess[j]
        dems.append([])
        xxs.append([])
        yys.append([])
        # Loop over the relevant DEMs
        for i in range(len(avail_files[j])):
            # Only load the NetCDF files
            if '.nc' in avail_fileDirs[j][i]:
                currentDEM = avail_fileDirs[j][i]
                # Load the netCDF file
                netCDFFile = nc.Dataset(currentDEM)
                # Extract the relevant information
                #pcs = np.array(netCDFFile.get_variables_by_attributes(name='Z percentiles')[0])
                zData = np.array(netCDFFile.get_variables_by_attributes(name='Z-axis')[0])
                
                # Meshgrid information is only necessary for first iteration
                if len(xxs[j]) == 0:
                    xAxis = np.array(netCDFFile.get_variables_by_attributes(name='X-axis')[0])
                    yAxis = np.array(netCDFFile.get_variables_by_attributes(name='Y-axis')[0])
                    # Convert the X/Y-axis information to plottable values
                    xValues = np.arange(xAxis[0],xAxis[1],xAxis[2])
                    yValues = np.arange(yAxis[0],yAxis[1],yAxis[2])
                    xx, yy = np.meshgrid(xValues,yValues)
                    xxs[j] = xx
                    yys[j] = yy
                    
                # We can close the file again
                netCDFFile.close()
                                
                # Transform zData to 2D grid
                zData =  zData[:,:,0].reshape((zData.shape[0],zData.shape[1]))
                
                # Append
                dems[j].append(zData)
                
                #####################################################
                # # Optionally visualize the DEM
                # fig = plt.figure(figsize=(30,4.5))
                # plt.pcolormesh(xxs[j], yys[j], zData,
                #                cmap="terrain",
                #                rasterized=True,
                #                vmax=0.1, 
                #                vmin=0.01)
                # plt.xlim(coords[0],coords[1])
                # plt.ylim(coords[2],coords[3])
                # plt.tight_layout()
                # plt.tick_params(labelsize=15)
                # plt.xlabel('x (m)', size=15)
                # plt.ylabel('y (m)', size=15)
                # plt.title(avail_files[j][i][:-3] + '   at   ' + demRes + ' mm   resolution', size=30)
                # # Add colorbar
                # clb = plt.colorbar(location='right', pad = 0.01)
                # clb.set_label('Z (m)', size=15)
                # clb.ax.tick_params(labelsize=15)
                #####################################################
    
    #%% Now conduct the desired analysis on the DEMs
    '''
    From here on you can write your own code to conduct desired analysis using the following variables:
    
    - demRess : list of computed resolutions (either 25, 5, or both)
    - dems : list of all DEMs within the experiment, separated by resolution
    - avail_files : associated filenames (usually including timestep (== cycle number for Exps052-078))
    - xxs, yys : associated meshgrids
                 Note because of different image coordinates, yys is vertically flipped compared to the NetCDF files of DEM and water depth maps
    
    '''
    
#%% How long did the run take?
end = datetime.datetime.now()
td = round((end - start).total_seconds())
print("Script running time = " + str(timedelta(seconds=td)))
