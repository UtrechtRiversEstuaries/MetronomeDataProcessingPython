"""
Python script that converts the raw laserscan data into a DEM

Python script written by Eise W. NOTA (finalized MAY 2025)

produces a structure g (gridded bathymetry data) that contains grids with units in m:
g.x and g.y: coordinates (m)
g.z50, g.zlow and g.zhigh: median elevation, low and high percentiles

Note: for making a complete laserscan of the metronome, the hard-coded software generally creates two additional useless files. 
This code moves these files into a bin in the raw folder. If the DEMs have been visually inspected to be correct, this bin can be manually deleted.

Note: for the calibration of the laser scanner, we refer to CalibrateLaserCamera.py and referenced scripts therein.
"""

#%% Clearing all variables
from IPython import get_ipython
get_ipython().magic('reset -sf') 

#%% Import packages
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.colors import LightSource
import shutil
import datetime
from datetime import timedelta
import netCDF4 as nc

# Set base directory of the network drive
pwd = r"\\..."

os.chdir(pwd + r"\python\Laserscan_scripts\Functions_laserscan") # Setting the functions directory
from Function_Laserscan_DEM import Compute_DEM
os.chdir(pwd + r"\python") # Setting the script directory
start = datetime.datetime.now()

#%% Set the baseModel you want to use
baseModel = '11'

#%% Defining the working directory and variables
Expnr = '054'                                    # Experiment number
pilot = 1                                        # Pilot number, 0 if no pilot
start_cycle = '00000'                            # String of cycle where you want to start # 00000 = first cycle
end_cycle = '01000'                              # String of cycle where you want to end # End with complete numbers as in folder structure

gridResols = [5]                                  # Desired grid resolutions in mm in list # Default is [5] #mm     
yCoordinates = [0,3]                             # Which range of y coordinates should the DEM contain? # Default [0,3]
xCoordinates = [0,20]                            # Which range of x coordinates should the DEM contain? # Default [0,20]
# Now which information do we want to store?
percentiles = [50]                               # Percentiles of elevations that we want. Default only 50, else add with commas
nrPoints = 0                                     # 1 if we want to store the number of points; 0 if we don't
average = 0                                      # 1 if you want to interpolate with the average instead of the median, in this case, set percentiles to [50]
domingCorrection = 1                             # 1 if we want to correct for doming (only when this is calibrated)

#%% Define the figure settings
figwidth = 30 # inches
figsize=(figwidth, figwidth*(3/20)) # scale y- and x-axes
labelsize = 15 # for labels x-, y-axes and colorbar
ticksize = 12.5 # for label ticks
titlesize = 20 # for title
zPlotRange = [0.01,0.10] # Which z values do you want to plot? #m
ls = LightSource(azdeg=225, altdeg=15) # Lightsource for hillshade
ve = 5 # Vertical exaggeration for hillshade
hillshadePlot = 1 # boolean, do we want to include hillshade in our plot?
colormeshPlot = 1 # boolean, do we want to include a colormesh without hillshade in our plot?
tightPlot = 1 # boolean, only pixels and no axes

#%% Defining the relevant folders
baseModelFolder = pwd + r"\python\Agisoft_BaseModel\BaseModel" + baseModel + "\ActualBaseModel"
calibFolder = pwd + r"\python\Calibration_scripts_and_parameters\CalibrationParameters\BaseModel" + baseModel
writefolders = []

if pilot == 0:                                                          
    rootfolder = pwd + r'\experiments\Exp' + Expnr + '\\raw_data\\laser_scanner'
    binfolder = pwd + r'\experiments\Exp' + Expnr + '\\raw_data\\laser_scanner_bin'
    for gridResol in gridResols:
        writefolders.append(pwd + r'\experiments\Exp' + Expnr + '\\processed_data\\DEMs\\laser_scanner\\BaseModel' + baseModel + '\\Res' + str(gridResol) + 'mm')
else:
    rootfolder = pwd + r'\experiments\Exp' + Expnr + '\Pilot' + str(pilot) + '\\raw_data\\laser_scanner'
    binfolder = pwd + r'\experiments\Exp' + Expnr + '\Pilot' + str(pilot) + '\\raw_data\\laser_scanner_bin'
    for gridResol in gridResols:
        writefolders.append(pwd + r'\experiments\Exp' + Expnr + '\Pilot' + str(pilot) + '\\processed_data\\DEMs\\laser_scanner\\BaseModel' + baseModel + '\\Res' + str(gridResol) + 'mm')

# Create writefolder if it doesn't exist
for writefolder in writefolders:
    if not os.path.exists(writefolder):
        os.makedirs(writefolder)

del(gridResol,writefolder)

#%% Loading input directories
available_folders = os.listdir(rootfolder) # Which sub folders are in the rootfolder?
chosen_folders = [] # Relevant folders will be stored here
      
# Create dictionary of relevant folders per camera based on start_cycle and end_cycle
for i in available_folders:
    # Only append if the folders are between start_cycle and end_cycle
    if i == start_cycle: # First start_cycle is discovered
        chosen_folders.append(i)
        # If start_cycle and end_cycle are the same, break the for-loop
        if start_cycle == end_cycle:
            break
    # Then subsequent folders are added
    elif len(chosen_folders) > 0 and i != end_cycle:
        chosen_folders.append(i)
    # Until finally the end_cycle is appended
    elif i == end_cycle: # First start_cycle is discovered
        chosen_folders.append(i)
        # End the for loop
        break

#%% Load the relevant calibration data
# For readability of the code, we import them here and take them as variables in the function
# Lens parameters of the laser camera to apply lens correction to the raw data
laserCamLens = pd.read_csv(calibFolder + r"\\laserCam_lens_parameters_final.csv",delimiter=',') # Extracted from BaseModel through ExtractCoAlignedCameraPositionsAndParameters.py
# The estimated trajectories of the positions and orientations of the laser line for all 20000 timesteps
laserPointerTr = pd.read_csv(calibFolder + r"\\laserPointer_trajectory_final.csv",delimiter=',') # Interpolated in InterpolateLaserCameraTrajectory.py
# The estimated trajectories of the positions and orientations of the laser camera for all 20000 timesteps
laserCamTr = pd.read_csv(calibFolder + r"\\laserCam_trajectory.csv",delimiter=',') # Interpolated in InterpolateLaserCameraTrajectory.py
# Doming correction, if applicable
if domingCorrection == 1:
    domingFunction = pd.read_csv(calibFolder + r"\\domingCorrectionFunction.csv",delimiter=',',header=1)
else:
    domingFunction = False

#%% Start the for-loop for computing the DEMs
for cycle in chosen_folders:
    # What is the current folder directory?
    currentFolder = rootfolder + '\\' + cycle
    # What are the files in the currentFolder?
    available_files = os.listdir(currentFolder)
    # Only tif files are relevant
    relevant_files = []
    relevant_files.append([file for file in available_files if file.endswith('.tif')])
    relevant_files = relevant_files[0] # Flatten list
    # Something is incorrect in the hard-coded laserscan software that it makes additional files we don't need
    # Each file is created 40 seconds after the previous one, wrong files are created quickly after the previous one
    # So the relevant information is in the timestamps of the files
    timeStamps = [] # Store the timeStamps
    for file in relevant_files:
        timeStamp = datetime.datetime.strptime(file[-27:-4],"%Y-%m-%d %H.%M.%S.%f") # Timestamps are always stored in this format with these indices
        timeStamps.append(timeStamp)
    # There may be an error in the first file if it stored at more than 40 seconds before the next file. Raise error
    if (timeStamps[1]-timeStamps[0]).total_seconds() > 80:
        raise ValueError('There is a >80 seconds time gap between first and second laserscan file of cycle ' + cycle + '. The first file is probably incorrect.')
    # If it's less than 45 seconds, we're assuming that the first file is correct.
    inputFiles = [currentFolder + '\\' + relevant_files[0]]
    # Now don't consider the files quickly created after the other. A limit of 5 seconds is a conservative estimate 
    for i in range(1,len(relevant_files)):
        if abs((timeStamps[i]-timeStamps[i-1]).total_seconds()) > 5:
            inputFiles.append(currentFolder + '\\' + relevant_files[i])
        else: # Move the unwated file to the binfolder
            # First test if the binfolder exists and make it if not
            if not os.path.exists(binfolder):
                os.makedirs(binfolder)
            # Move file to binfolder
            incorrectFile = '\\' + relevant_files[i]
            shutil.move(currentFolder + incorrectFile,binfolder + incorrectFile)
            del(incorrectFile)
    
    # Determine the file name of the DEM depending on whether it's a pilot, or the average is computed
    if pilot == 0:
        if average == 0:
            DEM_name = 'Exp' + Expnr + '_' + cycle + '_LaserScan_DEM.nc'
        else: # average == 1
            DEM_name = 'Exp' + Expnr + '_' + cycle + '_LaserScan_avg_DEM.nc'
    else:
        if average == 0:
            DEM_name = 'Exp' + Expnr + '_Pilot' + str(pilot) + '_' + cycle + '_LaserScan_DEM.nc'
        else: # average == 1
            DEM_name = 'Exp' + Expnr + '_Pilot' + str(pilot) + '_' + cycle + '_LaserScan_avg_DEM.nc'
    
    # Now that we have all the relevant information, we can compute the DEM.
    # Loop through the amount of gridResols
    for i in range(len(gridResols)):
        # Only compute if all DEMs don't already exist though.
        if not os.path.exists(writefolders[i] + '\\' + DEM_name):
            Compute_DEM(inputFiles,writefolders,DEM_name,gridResols,yCoordinates,xCoordinates,percentiles,nrPoints,
                        laserCamLens,laserPointerTr,laserCamTr,average,domingCorrection,domingFunction)
            # Only one computation is necessary; within the function it also tests whether a DEM already exists
            break
    
    #%% Now Visualize the DEMs and optionally nr of points
    for i in range(len(gridResols)):
        # Define the current writefolder and gridResol
        writefolder = writefolders[i]
        gridResol = gridResols[i]
        # Load the netCDF file
        netCDFFile = nc.Dataset(writefolder + '\\' + DEM_name)
        # Extract the relevant information
        xAxis = np.array(netCDFFile.get_variables_by_attributes(name='X-axis')[0])
        yAxis = np.array(netCDFFile.get_variables_by_attributes(name='Y-axis')[0])
        pcs = np.array(netCDFFile.get_variables_by_attributes(name='Z percentiles')[0])
        zData = np.array(netCDFFile.get_variables_by_attributes(name='Z-axis')[0])
        if nrPoints == 1:
            nps = np.array(netCDFFile.get_variables_by_attributes(name='Nr Points')[0])
        # We can close the file again
        netCDFFile.close()
        
        # Convert the X/Y-axis information to plottable values
        xValues = np.arange(xAxis[0],xAxis[1],xAxis[2])
        yValues = np.arange(yAxis[0],yAxis[1],yAxis[2])
        xx, yy = np.meshgrid(xValues,yValues)
        
        # Create the plots per percentile
        for i in range(len(pcs)):
            # What will be the figure name and title?
            # The following generates a pcolormesh without hillshade
            currentZ =  zData[:,:,i].reshape((zData.shape[0],zData.shape[1]))
            
            # If we want to make a hillshade
            if hillshadePlot == 1:
                # Define name and title
                if average == 0:
                    figName = DEM_name[:-3] + '_' + pcs[i] + '_hillshade.PNG'
                    figTitle = 'Exp' + Expnr + ' Cycle' + cycle + ' ' + pcs[i] + ' DEM Grid' + str(gridResol) + 'mm with hillshade'
                else: # average == 1
                    figName = DEM_name[:-3] + '_hillshade.PNG'
                    figTitle = 'Exp' + Expnr + ' Cycle' + cycle + ' average elevation DEM Grid' + str(gridResol) + 'mm with hillshade'
                
                # Make the plot
                fig = plt.figure(figsize=figsize)
                ax = fig.gca()
                rgb = ls.shade(currentZ, 
                               cmap=plt.cm.terrain,
                               vmax=zPlotRange[1], 
                               vmin=zPlotRange[0],
                               blend_mode='soft',
                               vert_exag=ve,
                               dx=xValues,
                               dy=yValues)
                ax.pcolormesh(xx, yy,rgb)
                plt.xlim(xCoordinates[0],xCoordinates[1])
                plt.ylim(yCoordinates[0],yCoordinates[1])
                plt.tight_layout()
                plt.tick_params(labelsize=ticksize)
                plt.xlabel('x (m)', size=labelsize)
                plt.ylabel('y (m)', size=labelsize)
                plt.title(figTitle, size=titlesize, pad=15)
                # We need a proxy artist for the colorbar...
                im = ax.imshow(currentZ,rasterized=True, cmap=plt.cm.terrain,vmax=zPlotRange[1],vmin=zPlotRange[0])
                im.remove()
                clb = fig.colorbar(im,ax=ax,location='right', pad = 0.01)
                clb.set_label('Z (m)', size=labelsize)
                clb.ax.tick_params(labelsize=ticksize)
                # Store image
                plt.savefig(writefolder + '\\' + figName, bbox_inches='tight')
                # Show image
                plt.show()
            
            # In case we just want a colormesh plot without hillshade
            if colormeshPlot == 1:
                # Define name and title
                if average == 0:
                    figName = DEM_name[:-3] + '_' + pcs[i] + '_colormesh.PNG'
                    figTitle = 'Exp' + Expnr + ' Cycle' + cycle + ' ' + pcs[i] + ' DEM Grid' + str(gridResol) + 'mm without hillshade'
                else: # average == 1
                    figName = DEM_name[:-3] + '_colormesh.PNG'
                    figTitle = 'Exp' + Expnr + ' Cycle' + cycle + ' average elevation DEM Grid' + str(gridResol) + 'mm without hillshade'
                    
                # Make the plot
                ax = plt.figure(figsize=figsize)
                plt.pcolormesh(xx, yy, currentZ,
                               cmap="terrain",
                               rasterized=True,
                               vmax=zPlotRange[1], 
                               vmin=zPlotRange[0])
                plt.xlim(xCoordinates[0],xCoordinates[1])
                plt.ylim(yCoordinates[0],yCoordinates[1])
                plt.tight_layout()
                plt.tick_params(labelsize=ticksize)
                plt.xlabel('x (m)', size=labelsize)
                plt.ylabel('y (m)', size=labelsize)
                plt.title(figTitle, size=titlesize, pad=15)
                # Add colorbar
                clb = plt.colorbar(location='right', pad = 0.01)
                clb.set_label('Z (m)', size=labelsize)
                clb.ax.tick_params(labelsize=ticksize)
                # Store image
                plt.savefig(writefolder + '\\' + figName, bbox_inches='tight')
                # Show image
                plt.show()
            
            # In case we just want a tightplot without axes, titles etc.
            if tightPlot == 1:
                # Define name
                if average == 0:
                    figName = DEM_name[:-3] + '_' + pcs[i] + '_tightmesh.PNG'
                else: # average == 1
                    figName = DEM_name[:-3] + '_tightmesh.PNG'
    
                # Make the plot
                #ax = plt.figure(figsize=figsize)
                fig = plt.figure()
                ax = fig.gca()
                im = ax.pcolormesh(xx, yy, currentZ,
                               cmap="terrain",
                               rasterized=True,
                               vmax=zPlotRange[1], 
                               vmin=zPlotRange[0]) 
                ax.axes.set_aspect("equal")
                #retrieve rgb values of the quadmesh object
                rgbIm = im.to_rgba(im.get_array(),bytes=True)[:,:,:3]
                # Use openCV to store, for this we need to make some adaptations
                # First flip
                rgbIm = np.flipud(rgbIm)
                # Isolate bands
                rBand = rgbIm[:,:,0]
                gBand = rgbIm[:,:,1]
                bBand = rgbIm[:,:,2]
                # Apply bands to openCV order
                bgrIm = np.zeros(shape=(len(yValues),len(xValues),3))
                bgrIm[:,:,0] = bBand
                bgrIm[:,:,1] = gBand
                bgrIm[:,:,2] = rBand
                cv2.imwrite(writefolder + '\\' + figName,bgrIm) # store debayered and corrected image in copyFolder
                plt.close()
        
        # Plot number of points in case we want them
        if nrPoints == 1:
            # Define name and title
            figName = DEM_name[:-3] + '_nrPoints.PNG'
            figTitle = 'Exp' + Expnr + ' Cycle' + cycle + ' DEM Grid' + str(gridResol) + 'mm number of points'
            # Make the plot
            ax = plt.figure(figsize=figsize)
            plt.pcolormesh(xx, yy, nps,
                           cmap="nipy_spectral",
                           rasterized=True,
                           vmax=np.percentile(nps,99), 
                           vmin=0)
            plt.xlim(xCoordinates[0],xCoordinates[1])
            plt.ylim(yCoordinates[0],yCoordinates[1])
            plt.tight_layout()
            plt.tick_params(labelsize=ticksize)
            plt.xlabel('x (m)', size=labelsize)
            plt.ylabel('y (m)', size=labelsize)
            plt.title(figTitle, size=titlesize, pad=15)
            # Add colorbar
            clb = plt.colorbar(location='right', pad = 0.01)
            clb.set_label('Z (m)', size=labelsize)
            clb.ax.tick_params(labelsize=ticksize)
            # Store image
            plt.savefig(writefolder + '\\' + figName, bbox_inches='tight')
            # Show image
            plt.show()
        # Extract the Z-data over the percentiles

#%% How long did the run take?
end = datetime.datetime.now()
td = round((end - start).total_seconds())
print("Script running time = " + str(timedelta(seconds=td)))