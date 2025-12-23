'''
Code to create a set amount of orthomosaics within an experiment from the DSLR imagery of the 
Metronome in the Earth Simulation Laboratory

Based on earlier code written by Brechtje A. van Amstel (SEPTEMBER 2023)
Eise W. NOTA (finalized JULY 2024)

Produces intrinsically and extrinsically corrected orthomosaics through co-alignment of a BaseModel from Overhead and DSLR imagery (Exp063_Cycle00000) 

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

# Set the agisoft_LICENSE environment variable
os.environ["..."] ='...' 
import Metashape

# Option to check if the license was loaded correctly
Metashape.license.valid

# Set base directory of the network drive
pwd = r"\\..."

# Import functions
os.chdir(pwd + r"\python\DSLR_scripts\Functions_DSLR") # Setting the functions directory
from Function_Metashape_DSLR import Orthomosaic_DSLR 
os.chdir(pwd + r"\python\DSLR_scripts")  # Setting the working directory to the folder with all codes & locations

start = datetime.datetime.now()

#%% Set the baseModel you want to use
baseModel = '11'
# For Exp052-066, the marker coordinates of BaseModel01 are more accurate than of BaseModel02 and beyond
# From Exp067 onwards, it's better to set the same markerModel as baseModel, before use BaseModel01
# Note: Exp058, testSurvey uses markerModel11
markerModel = '11'

#%% Defining the working directory and variables
Expnr = '058'                                                                   # Experiment number
pilot = 0                                                                       # Pilot number, 0 if no pilot
start_cycle = 'testSurvey'                                                           # String of cycle where you want to start, this folder will be included # No need to specify dry or wet
end_cycle = 'testSurvey'                                                             # String of cycle where you want to end, this folder will be included # No need to specify dry or wet
orthoRes = 5                                                                    # Desired image resolution in mm; on the windows server ~1:45 mins for 5 mm (default); ~3:30 minutes for 1 mm; ; ~6:00 minutes for 0.625 mm      
startXY = (0,0)                                                                 # At what x-coordinates do we want the orthomosiac to start? # Default (0,0)
endXY = (20,3)                                                                  # At what x-coordinates do we want the orthomosiac to end? # Default (20,3)

#%% Defining the root- and writefolder based on input variables
# Folder where the Agisoft BaseModel is stored
baseFolder = pwd + r"\python\Agisoft_BaseModel\BaseModel" + baseModel + "\ActualBaseModel"
rawPhotosBaseFolder = pwd + r"\python\Agisoft_BaseModel\BaseModel" + baseModel + "\Raw_photos"
# Folder with all the calibration parameters (as well as the Python scripts that derived them)
#calibrationFolder = pwd + r"\python\Calibration_scripts_and_parameters\CalibrationParameters\BaseModel" + baseModel
# Folder with the marker coordinates
markerFolder = pwd + r"\python\Calibration_scripts_and_parameters\CalibrationParameters\BaseModel" + markerModel

if pilot == 0:                                                          
    rootFolder = pwd + r'\experiments\Exp' + Expnr + '\\raw_data\\SLR_camera_laser_gantry'
    copyFolder = pwd + r'\experiments\Exp' + Expnr + '\\processed_data\\orthophotos\\copyBaseModel'
    copyRawFolder = pwd + r'\experiments\Exp' + Expnr + '\\processed_data\\orthophotos\\Raw_photos'
    writeFolder = pwd + r'\experiments\Exp' + Expnr + '\\processed_data\\orthophotos\\SLR_camera_laser_gantry\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\'
    namestart = 'Exp' + Expnr
else:
    rootFolder = pwd + r'\experiments\Exp' + Expnr + '\Pilot' + str(pilot) + '\\raw_data\\SLR_camera_laser_gantry'
    copyFolder = pwd + r'\experiments\Exp' + Expnr + '\Pilot' + str(pilot) + '\\processed_data\\copyBaseModel'
    copyRawFolder = pwd + r'\experiments\Exp' + Expnr + '\Pilot' + str(pilot) + '\\processed_data\\Raw_photos'
    writeFolder = pwd + r'\experiments\Exp' + Expnr + '\Pilot' + str(pilot) + '\\processed_data\\orthophotos\\SLR_camera_laser_gantry\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\'
    namestart = 'Exp' + Expnr + '_Pilot' + str(pilot)

# Create/copy a directory if it does not exist
if not os.path.exists(writeFolder):
    os.makedirs(writeFolder)    
    
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
    
#%% Load calibration parameters
# Computing a correct orthomosaic requires some (pre-)processing steps, which have been determined in previous calibration analyses 
# These calibration parameters have been determined in either Python codes or the Agisoft BaseModel
# We load these files in the main code, instead of in the separate functions, to enhance the understandibility of the codes

# Load marker coordinates of the DSLR images, as determined in the Agisoft BaseModel
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

# Also load the positions and orientations of the cameras
#camPos = pd.read_csv(calibrationFolder + r"\\DSLR_positions_orientations.csv",delimiter=',')

#%% Defining which pictures will be created based on start- and end-cycle
available_folders = os.listdir(rootFolder)

# In case cycles can be included as integers (default)
chosen_folders = []
for folder in available_folders:
    if int(start_cycle) <= int(folder[:5]) <= int(end_cycle):                   # Only folders are selected that are between defined start and end cycles
        chosen_folders.append(folder)
    
# Enable below line instead if deviating from default
#chosen_folders = [start_cycle]    
    
#%% Creating orthomosaics through Metashape
# Loop through the chosen_folders
for folder in chosen_folders:
    print("Current photo sequence: " + folder)  
    # Determine the image_name
    image_name = namestart + '_' + folder +'.JPG'
    # Store directories of the separate input_files
    # Create the Orthomosaic if it doesn't already exist in the orthoFolder
    if not os.path.exists(writeFolder + image_name):
        input_files = glob.glob(rootFolder + '\\' + folder + '\\*.JPG', recursive=True)
        # Compute the orthomosaic
        Orthomosaic_DSLR(input_files, baseModel, writeFolder, copyFolder, image_name, orthoRes, startXY, endXY, markerCSV) # , camPos)
       
# How long did the run take?
end = datetime.datetime.now()
td = round((end - start).total_seconds())
print("Script running time = " + str(datetime.timedelta(seconds=td)))
