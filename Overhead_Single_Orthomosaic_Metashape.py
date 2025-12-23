'''
Code to create a single RGB overhead orthomosaic from the Overhead cameras of the 
Metronome in the Earth Simulation Laboratory

Python script written by Eise W. NOTA (finalized JUNE 2024)

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

# Set the agisoft_LICENSE environment variable
os.environ["..."] ='...' 
import Metashape

# Option to check if the license was loaded correctly
Metashape.license.valid

# Set base directory of the network drive
pwd = r"\\..."

# Import functions
os.chdir(pwd + r"\python\Overhead_scripts\Functions_Overhead") # Setting the functions directory
from Function_Metashape_Overhead import Orthomosaic_Overhead 
os.chdir(pwd + r"\python\Overhead_scripts")  # Setting the working directory to the folder with all codes & locations

start = datetime.datetime.now()

#%% Set the baseModel you want to use
baseModel = '11'

#%% Define the variables
Expnr = '054'                   # Experiment number
pilot = 1                       # Pilot number, 0 if no pilot
cycle = '09999'                 # String of cycle you want to compute # Note this is Python counting, so 0 is first, 1 second etc.
cycleFolder = r'00500-01000'    # String of cycle folder in which the cycle is                         
orthoRes = 5                    # Desired image resolution in mm; 5 is the default low resolution; 3 is the default high resolution (higher is possible, but doesn't work for the sea side)         
tilting_imagery = False         # False in case there is only one photo per cycle at horizontal position; else True
startXY = (0,0)                 # At what x-coordinates do we want the orthomosiac to start? # Default (0,0)
endXY = (20,3)                  # At what x-coordinates do we want the orthomosiac to end? # Default (20,3)

# Which index does the cycle have in the cycleFolder?
cycleIndex = int(cycle)-int(cycleFolder[0:5]) # Possibly this would require manual change if folder name deviates from standard

# Note that Exp058 has a very specific directory setup deviating from our standard directory layout, so it requires additional coding
# If you want to compute overhead orthomosaics of Exp058, the following code needs to be rewritten: \\ad.geo.uu.nl\FG\River Delta Land\Metronome\python\Archive_Complete\Stitch_Overhead_Metashape

#%% Defining the root- and writefolder based on input variables
# Folder where the Agisoft BaseModel is stored
baseFolder = pwd + r"\python\Agisoft_BaseModel\BaseModel" + baseModel + "\ActualBaseModel"
rawPhotosBaseFolder = pwd + r"\python\Agisoft_BaseModel\BaseModel" + baseModel + "\Raw_photos"
calibrationFolder = pwd + r"\python\Calibration_scripts_and_parameters\CalibrationParameters\BaseModel" + baseModel

# Defining the root- and writefolders based on the experiment and setting
if pilot == 0:                                                          
    rootFolder = pwd + r'\experiments\Exp' + Expnr + '\\raw_data\\overhead_cameras'
    copyFolder = pwd + r'\experiments\Exp' + Expnr + '\\processed_data\\orthophotos\\copyBaseModel'
    copyRawFolder = pwd + r'\experiments\Exp' + Expnr + '\\processed_data\\orthophotos\\Raw_photos'
    writeFolder = pwd + r'\experiments\Exp' + Expnr + '\\processed_data\\orthophotos\\overhead_cameras\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\'
    image_name = 'Exp' + Expnr + '_RGB_' + cycle + '.JPG' 
else:
    rootFolder = pwd + r'\experiments\Exp' + Expnr + '\Pilot' + str(pilot) + '\\raw_data\\overhead_cameras'
    copyFolder = pwd + r'\experiments\Exp' + Expnr + '\Pilot' + str(pilot) + '\\processed_data\\copyBaseModel'
    copyRawFolder = pwd + r'\experiments\Exp' + Expnr + '\Pilot' + str(pilot) + '\\processed_data\\Raw_photos'
    writeFolder = pwd + r'\experiments\Exp' + Expnr + '\Pilot' + str(pilot) + '\\processed_data\\orthophotos\\overhead_cameras\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\'
    image_name = 'Exp' + Expnr + '_Pilot' + str(pilot) + '_RGB_' + cycle + '.JPG'
    
# Create a directory if it does not exist
if not os.path.exists(writeFolder):
    os.makedirs(writeFolder)    
    
if not os.path.exists(copyFolder):
    shutil.copytree(baseFolder, copyFolder) 

# If the copydirectory exists, test it contains the proper BaseModel
if not os.path.exists(copyFolder + "\\AgisoftBaseModel" + baseModel + ".psx"):
    raise Exception("CopyFolder contains incorrect BaseModel. Delete the copyBaseModel and Raw_photos folders.")
    
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

# All of the overhead cameras have their own color deviations, so initial white balancing based on the DSLR survey in the BaseModel is required
initialWBCSV = pd.read_csv(calibrationFolder + r"\\Initial_white_balancing.csv",delimiter=',',header=None) # Calibrated with CalibrateOverheadCameraColors.py

# Load marker coordinates of the Overheads, as determined in the Agisoft BaseModel # Only possible if tilting_imagery = False
if tilting_imagery == False:
    markerCSV = pd.read_csv(calibrationFolder + r"\\Overhead_markers_positions.csv",delimiter=',')
else:
    markerCSV = False

#%% Defining which pictures will be created based on cycle and cycleFolder
# store directory with list of files
input_files = [
    glob.glob(rootFolder + '\cam_1\\' + cycleFolder + '\\' + '1_' + '*.bmp', recursive=True)[cycleIndex],
    glob.glob(rootFolder + '\cam_2\\' + cycleFolder + '\\' + '2_' + '*.bmp', recursive=True)[cycleIndex],
    glob.glob(rootFolder + '\cam_3\\' + cycleFolder + '\\' + '3_' + '*.bmp', recursive=True)[cycleIndex],
    glob.glob(rootFolder + '\cam_4\\' + cycleFolder + '\\' + '4_' + '*.bmp', recursive=True)[cycleIndex],
    glob.glob(rootFolder + '\cam_5\\' + cycleFolder + '\\' + '5_' + '*.bmp', recursive=True)[cycleIndex],
    glob.glob(rootFolder + '\cam_6\\' + cycleFolder + '\\' + '6_' + '*.bmp', recursive=True)[cycleIndex],
    glob.glob(rootFolder + '\cam_7\\' + cycleFolder + '\\' + '7_' + '*.bmp', recursive=True)[cycleIndex]
]

#%% Creating orthomosaic through Metashape
Orthomosaic_Overhead(input_files, baseModel, writeFolder, copyFolder, image_name, initialWBCSV, orthoRes, startXY, endXY, markerCSV, tilting_imagery) # white_balancing_pixel?

# How long took the run?
end = datetime.datetime.now()
td = round((end - start).total_seconds())
print("Script running time = " + str(datetime.timedelta(seconds=td)))
