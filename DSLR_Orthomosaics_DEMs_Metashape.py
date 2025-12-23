'''
Code to create a set amount of orthomosaics and DEMS within an experiment from the DSLR imagery of the 
Metronome in the Earth Simulation Laboratory

This code has the same structure as DSLR_Orthomosaics_Metashape.py,which only computes orthomosaics and no DEMs.
Eise W. NOTA (finalized SEPTEMBER 2024)

Produces intrinsically and extrinsically corrected orthomosaics and DEMs through co-alignment of a Metronome BaseModel.
The DEMs produced by this code are not generated from the laserscanner data, for this run Laserscan2DEM.py.

Note: the Metronome has cartesian coordinates: 
x = 0-20 m along the flume to the right
y = 0-3 m across the flume to the back wall
z = 0-0.4 m elevation above the flume floor
'''

#%% Clearing all variables
from IPython import get_ipython
get_ipython().magic('reset -sf') 

#%% Import packages
import datetime
import shutil
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.colors import LightSource
import netCDF4 as nc

# Set the agisoft_LICENSE environment variable
os.environ["..."] = '...' 
import Metashape

# Option to check if the license was loaded correctly
Metashape.license.valid

# Set base directory of the network drive
pwd = r"\\..."

# Import functions
os.chdir(pwd + r"\python\DSLR_scripts\Functions_DSLR") # Setting the functions directory
from Function_Metashape_DSLR import Orthomosaic_DEM_DSLR
from Function_Metashape_DSLR import Orthomosaic_DEM_DSLR_indv
from Function_GeoTIFF2NetCDF import geoTIFF2NetCDF
os.chdir(pwd + r"\python\DSLR_scripts")  # Setting the working directory to the folder with all codes & locations

start = datetime.datetime.now()

#%% Set the baseModel you want to use in string (e.g.: '11'), if you want to conduct individual alignment, set to False
baseModel = '11'
# In case the baseModel is not False, we'll assign GCP coordinates with the marker model
# For Exp052-066, the marker coordinates of BaseModel01 are more accurate than of BaseModel02 and beyond
# Note: Exp058, testSurvey uses markerModel11
markerModel = '11'

if baseModel == False: # which dummy Model will we use for the global positions of the markers? E.g. '11'
    dummyModel = '11'
    
#%% Defining the working directory and variables
Expnr = '058'                                                                   # Experiment number
pilot = 0                                                                       # Pilot number, 0 if no pilot
start_cycle = 'testSurvey'                                                           # String of cycle where you want to start, this folder will be included # No need to specify dry or wet
end_cycle = 'testSurvey'                                                             # String of cycle where you want to end, this folder will be included # No need to specify dry or wet
gridRes = 5                                                                     # Desired image/DEM  resolution in mm; on the windows server ~1:45 mins for 5 mm (default); ~3:30 minutes for 1 mm; ; ~6:00 minutes for 0.625 mm      
startXY = (0,0)                                                                 # At what x-coordinates do we want the orthomosiac to start? # Default (0,0)
endXY = (20,3)                                                                  # At what x-coordinates do we want the orthomosiac to end? # Default (20,3)
skipWet = 1                                                                     # Set to 1 if we want to skip wet DSLR surveys
depthMapSetting = 'medium'                                                      # 'lowest' / 'low' / 'medium' / 'high' / 'ultra'

#%% Define the DEM figure settings
yCoordinates = [startXY[1],endXY[1]]
xCoordinates = [startXY[0],endXY[0]]
figwidth = 30 # inches
figsize=(figwidth, figwidth*(3/20)) # scale y- and x-axes
labelsize = 15 # for labels x-, y-axes and colorbar
ticksize = 12.5 # for label ticks
titlesize = 20 # for title
zPlotRange = [0.01,0.1] # Which z values do you want to plot?  
ls = LightSource(azdeg=225, altdeg=15) # Lightsource for hillshade
ve = 5 # Vertical exaggeration for hillshade
hillshadePlot = 1 # boolean, do we want to include hillshade in our plot?
colormeshPlot = 1 # boolean, do we want to include a colormesh without hillshade in our plot?
tightPlot = 1 # boolean, only pixels and no axes

#%% Defining the root- and writefolder based on input variables
# Some folders depend on whether we want to use a BaseModel
if baseModel != False:
    # Folder where the Agisoft BaseModel is stored
    baseFolder = pwd + r"\python\Agisoft_BaseModel\BaseModel" + baseModel + "\ActualBaseModel"
    rawPhotosBaseFolder = pwd + r"\python\Agisoft_BaseModel\BaseModel" + baseModel + "\Raw_photos"
    # Pilot?
    if pilot == 0:                                                          
        copyFolder = pwd + r'\experiments\Exp' + Expnr + '\\processed_data\\orthophotos\\copyBaseModel'
        copyRawFolder = pwd + r'\experiments\Exp' + Expnr + '\\processed_data\\orthophotos\\Raw_photos'
        writeOrthoFolder = pwd + r'\experiments\Exp' + Expnr + '\\processed_data\\orthophotos\\SLR_camera_laser_gantry\\BaseModel' + baseModel + '\\Res' + str(gridRes) + 'mm\\'
        writeDEMFolder = pwd + r'\experiments\Exp' + Expnr + '\\processed_data\\DEMs\\Agisoft\\BaseModel' + baseModel + '\\Res' + str(gridRes) + 'mm\\'
    else:
        copyFolder = pwd + r'\experiments\Exp' + Expnr + '\Pilot' + str(pilot) + '\\processed_data\\copyBaseModel'
        copyRawFolder = pwd + r'\experiments\Exp' + Expnr + '\Pilot' + str(pilot) + '\\processed_data\\Raw_photos'
        writeOrthoFolder = pwd + r'\experiments\Exp' + Expnr + '\Pilot' + str(pilot) + '\\processed_data\\orthophotos\\SLR_camera_laser_gantry\\BaseModel' + baseModel + '\\Res' + str(gridRes) + 'mm\\'
        writeDEMFolder = pwd + r'\experiments\Exp' + Expnr + '\Pilot' + str(pilot) + '\\processed_data\\DEMs\\Agisoft\\BaseModel' + baseModel + '\\Res' + str(gridRes) + 'mm\\'
    
    # Create/copy a directory if it does not exist        
    if not os.path.exists(copyFolder):
        shutil.copytree(baseFolder, copyFolder)

    # If the copydirectory exists, test it contains the proper BaseModel
    if not os.path.exists(copyFolder + "\\AgisoftBaseModel" + baseModel + ".psx"):
        raise Exception("CopyFolder contains incorrect BaseModel. Delete copyBaseModel and Raw_photos folders.")

    # In the case of the base Model and raw photos, only overhead and DSLR are required
    if not os.path.exists(copyRawFolder + r'\\Overhead_calibrated'):
        shutil.copytree(rawPhotosBaseFolder + r'\\Overhead_calibrated', copyRawFolder + r'\\Overhead_calibrated') 

    if not os.path.exists(copyRawFolder + r'\\SLR_camera_laser_gantry'):
        shutil.copytree(rawPhotosBaseFolder + r'\\SLR_camera_laser_gantry', copyRawFolder + r'\\SLR_camera_laser_gantry') 

    if not os.path.exists(copyRawFolder + r'\\Laser_camera'):
        shutil.copytree(rawPhotosBaseFolder + r'\\Laser_camera', copyRawFolder + r'\\Laser_camera')
        
else: # Individual alignment
    # Marker positions location
    markerPosFolder = pwd + r'\python\Calibration_scripts_and_parameters\CalibrationParameters\BaseModel' + dummyModel
    # Pilot?
    if pilot == 0:
        writeOrthoFolder = pwd + r'\experiments\Exp' + Expnr + '\\processed_data\\orthophotos\\SLR_camera_laser_gantry\\IndividualAlignment\\Res' + str(gridRes) + 'mm\\'
        writeDEMFolder = pwd + r'\experiments\Exp' + Expnr + '\\processed_data\\DEMs\\Agisoft\\IndividualAlignment\\Res' + str(gridRes) + 'mm\\'
    else:
        writeOrthoFolder = pwd + r'\experiments\Exp' + Expnr + '\Pilot' + str(pilot) + '\\processed_data\\orthophotos\\SLR_camera_laser_gantry\\IndividualAlignment\\Res' + str(gridRes) + 'mm\\'
        writeDEMFolder = pwd + r'\experiments\Exp' + Expnr + '\Pilot' + str(pilot) + '\\processed_data\\DEMs\\Agisoft\\IndividualAlignment\\Res' + str(gridRes) + 'mm\\'
    
# Define conditions, regardless of whether we want to use a BaseModel 
# Folder with the marker coordinates
markerFolder = pwd + r"\python\Calibration_scripts_and_parameters\CalibrationParameters\BaseModel" + markerModel
# rootFolder and namestart
if pilot == 0:                                                          
    rootFolder = pwd + r'\experiments\Exp' + Expnr + '\\raw_data\\SLR_camera_laser_gantry'
    namestart = 'Exp' + Expnr
else:
    rootFolder = pwd + r'\experiments\Exp' + Expnr + '\Pilot' + str(pilot) + '\\raw_data\\SLR_camera_laser_gantry'
    namestart = 'Exp' + Expnr + '_Pilot' + str(pilot)

# Create/copy a directory if it does not exist
if not os.path.exists(writeOrthoFolder):
    os.makedirs(writeOrthoFolder)    

if not os.path.exists(writeDEMFolder):
    os.makedirs(writeDEMFolder)  
    
#%% Load calibration parameters
# Computing a correct orthomosaic requires some (pre-)processing steps, which have been determined in previous calibration analyses 
# These calibration parameters have been determined in either Python codes or the Agisoft BaseModel
# We load these files in the main code, instead of in the separate functions, to enhance the understandibility of the codes

# Load marker coordinates of the DSLR images, as determined in the marker BaseModel
markerCSV = pd.read_csv(markerFolder + r"\\DSLR_markers_positions.csv",delimiter=',')

# From BaseModel02 there are 81 DSLR images, while a normal survey is 41 surveys
# In case the markermodel isn't 01, we need to correct for this in case both surveys are not equal
if markerModel != '01':
    # To reduce markerCSV, we have to get rid of all uneven camindices
    for i in range(1,np.size(markerCSV,0)):
        if markerCSV.at[i,"camindex"] % 2 != 0: # Value is uneven
            # Delete
            markerCSV = markerCSV.drop(i)
        else: # Value is even
            # Divide by 2 to get new index
            markerCSV.at[i,"camindex"] = markerCSV.at[i,"camindex"]/2
    # Reset indices
    markerCSV = markerCSV.reset_index()

# In case we don't use the BaseModel, load DSLR positions from the dummy model
if baseModel == False:
    markerPos = pd.read_csv(markerPosFolder + r"\\marker_positions.csv",delimiter=',')

#%% Defining which pictures will be created based on start- and end-cycle
available_folders = os.listdir(rootFolder)

chosen_folders = []
condition = 0
for folder in available_folders:
    # Test if start_cycle occurs
    # We'll also test [:5] to ignore _dry and _wet
    if folder[:5] == start_cycle or folder == start_cycle:
        condition = 1 # change condition
    # Append if condition is met
    if condition == 1:                   # Only folders are selected that are between defined start and end cycles
        chosen_folders.append(folder)
    # Test if end_cycle occurs
    if folder[:5] == end_cycle or folder == start_cycle:
        condition = 0 # change condition back to 0

# Remove the wet folders if we want to skip these for making DEMs
if skipWet == 1:
    chosen_folders1 = []
    for folder in chosen_folders:
        #print(folder)
        if 'wet' not in folder:                   # Only folders that don't contain 'wet'
            chosen_folders1.append(folder)
    # Update chosen_folders
    chosen_folders = chosen_folders1; del(chosen_folders1)
            
#%% Creating orthomosaics through Metashape
# Loop through the chosen_folders
for folder in chosen_folders:
    print("Current photo sequence: " + folder)  
    # Determine the image_name
    ortho_name = namestart + '_' + folder + '_' + depthMapSetting +'.JPG'
    # Determine the DEM_names for the TIFF and netCDF files
    demTif_name = namestart + '_' + folder + '_' + depthMapSetting + '_Agisoft_DEM_GeoTIFF.TIF'
    demOrtho_name = namestart + '_' + folder + '_' + depthMapSetting + '_Agisoft_DEM_Ortho.JPG'
    demNetCDF_name = namestart + '_' + folder + '_' + depthMapSetting + '_Agisoft_DEM.nc'
    
    # Store directories of the separate input_files
    # Create the Orthomosaic + DEM + DEM ortho if it doesn't already exist in the orthoFolder
    if not os.path.exists(writeDEMFolder + demTif_name):
        input_files = glob.glob(rootFolder + '\\' + folder + '\\*.JPG', recursive=True)
        # Compute the orthomosaic and DEM, depending on whether we use a basemodel or not
        if baseModel != False:
            Orthomosaic_DEM_DSLR(input_files, baseModel, writeOrthoFolder, writeDEMFolder, copyFolder, ortho_name, demTif_name, demOrtho_name, gridRes, startXY, endXY, markerCSV, depthMapSetting)
        else: # Individual alignment
            Orthomosaic_DEM_DSLR_indv(input_files, writeOrthoFolder, writeDEMFolder, ortho_name, demTif_name, demOrtho_name, gridRes, startXY, endXY, markerCSV, markerPos, depthMapSetting)
    
    # Translate the GeoTIFF into a netCDF DEM with the same format as the laserscanDEM
    # Only if the DEM doesn't exist
    if not os.path.exists(writeDEMFolder + demNetCDF_name):
        # Run the function
        geoTIFF2NetCDF(writeDEMFolder, demTif_name, demNetCDF_name, gridRes, startXY, endXY)
    
    #%% Now Visualize the DEM and
    # Load the netCDF file
    netCDFFile = nc.Dataset(writeDEMFolder + demNetCDF_name)
    # Extract the relevant information
    xAxis = np.array(netCDFFile.get_variables_by_attributes(name='X-axis')[0])
    yAxis = np.array(netCDFFile.get_variables_by_attributes(name='Y-axis')[0])
    zData = np.array(netCDFFile.get_variables_by_attributes(name='Z-axis')[0])
    # We can close the file again
    netCDFFile.close()
    
    # Convert the X/Y-axis information to plottable values
    xValues = np.arange(xAxis[0],xAxis[1],xAxis[2])
    yValues = np.arange(yAxis[0],yAxis[1],yAxis[2])
    xx, yy = np.meshgrid(xValues,yValues)
    
    # Create the required plots
    # If we want to make a hillshade
    if hillshadePlot == 1:
        # Define name and title
        figName = demNetCDF_name[:-3] + '_hillshade.PNG'
        figTitle = 'Exp' + Expnr + ' Cycle' + folder + ' Agisoft DEM Grid' + str(gridRes) + 'mm with hillshade'
        # Make the plot
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
        rgb = ls.shade(zData, 
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
        im = ax.imshow(zData,rasterized=True, cmap=plt.cm.terrain,vmax=zPlotRange[1],vmin=zPlotRange[0])
        im.remove()
        clb = fig.colorbar(im,ax=ax,location='right', pad = 0.01)
        clb.set_label('Z (m)', size=labelsize)
        clb.ax.tick_params(labelsize=ticksize)
        # Store image
        plt.savefig(writeDEMFolder + figName, bbox_inches='tight')
        # Show image
        plt.show()
    
    # In case we just want a colormesh plot without hillshade
    if hillshadePlot == 1:
        # Define name and title
        figName = demNetCDF_name[:-3] + '_colormesh.PNG'
        figTitle = 'Exp' + Expnr + ' Cycle' + folder + ' Agisoft DEM Grid' + str(gridRes) + 'mm without hillshade'
        # Make the plot
        ax = plt.figure(figsize=figsize)
        plt.pcolormesh(xx, yy, zData,
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
        plt.savefig(writeDEMFolder + figName, bbox_inches='tight')
        # Show image
        plt.show()
    
    # In case we just want a tightplot without axes, titles etc.
    if tightPlot == 1:
        # Define name and title
        figName = demNetCDF_name[:-3] + '_tightmesh.PNG'
        # Make the plot
        #ax = plt.figure(figsize=figsize)
        fig = plt.figure()
        ax = fig.gca()
        im = ax.pcolormesh(xx, yy, zData,
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
        cv2.imwrite(writeDEMFolder + figName,bgrIm) # store debayered and corrected image in copyFolder
        plt.close()
    
# How long did the run take?
end = datetime.datetime.now()
td = round((end - start).total_seconds())
print("Script running time = " + str(datetime.timedelta(seconds=td)))

# Run times for a single DEM without (Exp057, 00000) & with (Exp057, 03750) developed morphology and a resolution of 5 mm
# 'lowest' >> Script running time >> 0:03:08 - 0:03:20
# 'low'    >> Script running time >> 0:03:58 - 0:04:04
# 'medium' >> Script running time >> 0:07:25 - 0:07:34
# 'high'   >> Script running time >> 0:21:10 - 0:21:27
# 'ultra'  >> Script running time >> 1:20:56 - 1:19:42