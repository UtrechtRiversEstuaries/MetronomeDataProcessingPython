'''
Demo code to load interval orthomosaics from the dataset of experimental estuaries (Nota et al., 2026)

Note that all experimental metadata is loaded from a separate file. 
Accordingly, only the indices of the experiments of interest need to be set in this code (line 45)

Furthermore, the desired resolutions need to be set, either '5'; '25'; or both (line 35)

Author: Eise W. Nota (finalized APRIL 2026)   

'''

#%% Clearing all variables
from IPython import get_ipython
get_ipython().magic('reset -sf') 

#%% Import packages
import os
import pandas as pd
import datetime
import cv2
import numpy as np
from datetime import timedelta
# For optional plotting:
import matplotlib.pyplot as plt
from pandas import Timedelta

#%% Set base directory of the dataset network drive (here set to align with the folder structure in the repository)
pwd = r"\\...\original"

# Set the directory of the functions
start = datetime.datetime.now()

#%% Set desired resolutions
orthoRess = ['25','5']  # List of desired image resolutions in mm; options: '25'; '5'

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
    
    # Pass iteration if there are no water depth maps
    if int(end_interval) - int(start_interval) == 0:
        continue
        
    # Compute meshgrids for set resolutions
    xxs = []
    yys = []
    for orthoRes in orthoRess:
        orthoRes = float(orthoRes)/1000 # from mm to m
        xValues = np.arange(coords[0]+orthoRes/2,coords[1]+orthoRes/2,orthoRes)
        yValues = np.arange(coords[2]+orthoRes/2,coords[3]+orthoRes/2,orthoRes)
        xx, yy = np.meshgrid(xValues,yValues)
        # Because image coordinates start in the top leftt, and the Metronome coordinate system starts in the lower left, yy needs to be flipped
        yy = np.flipud(yy)
        xxs.append(xx)
        yys.append(yy)
    
    # Define label start for relevant folders
    # Experimental folder
    if pilot == 0:
        expStart = 'Exp' + Expnr
    else:
        expStart = 'Exp' + Expnr + '_Pilot' + str(pilot)
    # Define appropriate indices
    expIndex = [expFolders.index(l) for l in expFolders if l.startswith(expStart)][0]
    
    #%% Define the relevant folders for mask and Water Depth Maps
    # Current experiment folder
    expFolder = pwd + '\\metronome_experiments\\' + expFolders[expIndex]
    
    # Add subsequent folders
    orthoFolders = []
    for orthoRes in orthoRess:
        orthoFolders.append(expFolder + '\\Orthophotos\\Overhead Cameras\\BaseModel' + baseModel + '\\Res' + orthoRes + 'mm')
        
    #%% Define the relevant orthos and their information
    avail_files = []
    avail_fileDirs = []
    # Create list of available orthos
    for orthoFolder in orthoFolders:
        avail_files.append(os.listdir(orthoFolder))
        avail_fileDirs.append([os.path.join(orthoFolder, file) for file in os.listdir(orthoFolder)])
    
    # Determine relevant depth maps
    rel_orthos = []
    rel_timesteps = []
    # Loop through the ortho resolutions
    for j in range(len(orthoRess)):
        orthoRes = orthoRess[j]
        rel_orthos.append([])
        rel_timesteps.append([])
        # Loop through available files
        for i in range(len(avail_files[j])):
            file = avail_files[j][i]
            # First test whether the cycles occur within an irrelevant range
            if file[-9:-4] not in cyclesToSkip:
                if int(start_interval) <= int(file[-9:-4]) <= int(end_interval): 
                    rel_orthos[j].append(avail_fileDirs[j][i])
                    rel_timesteps[j].append(int(file[-9:-4]))
    
    # Make timestamps 
    rel_timestamps = []
    for j in range(len(orthoRess)):
        rel_timestamps.append([])
        rel_timestep = np.array(rel_timesteps[j])
        rel_timestamp = rel_timestep * cycleDurat # seconds
        # Sort
        rel_orthos[j] = sorted(rel_orthos[j], key=lambda x: x.split('_')[2])
        rel_timesteps[j] = sorted(rel_timestep)
        rel_timestamps[j] = sorted(rel_timestamp)

    # Do a final filtering on cyclesRangesToSkip, in case they are set
    if len(cyclesRangesToSkip[0]) != 0:
        for j in range(len(orthoRess)):
            # Reversed for loop to maintain proper indices
            for i in range(len(rel_timesteps[j])-1,-1,-1):
                for currentRange in cyclesRangesToSkip:
                    if int(currentRange[:5]) <= rel_timesteps[j][i] < int(currentRange[-5:]):
                        rel_timesteps[j].pop(i)
                        rel_orthos[j].pop(i)
                        rel_timestamps[j].pop(i)
        
    #%% Load all depth Maps into a single list
    orthos = []
    # Loop over the relevant resolutions
    for j in range(len(rel_orthos)):
        orthoRes = orthoRess[j]
        orthos.append([])
        # Loop over the relevant orthos
        for i in range(len(rel_orthos[j])):
            currentOrtho = rel_orthos[j][i]
            # Load currentOrtho
            ortho = cv2.imread(currentOrtho)
            # Append
            orthos[j].append(ortho)
            
            #####################################################
            # # Optionally visualize the ortho
            # # Note that we manually change the coordinate labels as imshow plots using pixel coordinates
            # h, w = ortho.shape[:2]
            # ypixelstep = h/3 # Divide by 3 to plot each metre along y-axis
            # xpixelstep = w/10 # Divide by 10 to plot each 2 metres along y-axis
            # fig, ax = plt.subplots(figsize=(30,4.5),layout='constrained')
            # ax.imshow(ortho[:, :, ::-1],interpolation=None)
            # ax.set_xticks(np.arange(0,w+xpixelstep,xpixelstep))
            # ax.set_xticklabels(['0','2','4','6','8','10','12','14','16','18','20 m'],fontsize=15)
            # ax.set_yticks(np.arange(0,h+ypixelstep,ypixelstep))
            # ax.set_yticklabels(['3 m','2','1','0'],fontsize=15)
            # ax.set_title(expStart + '   Timestep ' + "{0:05d}".format(rel_timesteps[j][i]) + '   (' + str(Timedelta(seconds=rel_timestamps[j][i])) + ')' + '   at   ' + orthoRes + ' mm   resolution', size=30)
            #####################################################
     
    #%% Now conduct the desired analysis on the orthomosaics
    '''
    From here on you can write your own code to conduct desired analysis using the following variables:
    
    - orthoRess : list of computed resolutions (either 25, 5, or both)
    - orthos : list of all orthos within the experiment, separated by resolution
    - rel_timesteps : associated timesteps (these equal tidal cycle numbers for Exp052-Exp078)
    - rel_timestamps : associated timestamps in seconds from the beginning of the experiment
    - xxs, yys : associated meshgrids
                 Note because of different image coordinates, yys is vertically flipped compared to the NetCDF files of DEMs and water depth maps
    
    '''
    
#%% How long did the run take?
end = datetime.datetime.now()
td = round((end - start).total_seconds())
print("Script running time = " + str(timedelta(seconds=td)))
