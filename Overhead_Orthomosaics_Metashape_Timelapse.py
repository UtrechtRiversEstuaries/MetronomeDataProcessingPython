'''
Code to create overhead orthomosaics and movies from the overhead cameras of the 
Metronome in the Earth Simulation Laboratory
The frames are created through generating orthomosaics from the overhead imagery using Agisoft

Earlier codes have methods to deal with disfunctioning cameras, too many images or too little images.
In this code, I deliberately did not incorporate these, as the cameras generally work fine during the experiments since the end of 2022.
Moreover, the solving is conducted in a way which makes output not applicable for reliable analysis.

Note: the Metronome has cartesian coordinates: 
x = 0-20 m along the flume to the right
y = 0-3 m across the flume to the back wall
z = 0-0.4 m elevation above the flume floor

Note: To run this script you need to install packages from the environment.yml:
    - Create the Metronome environment on your user drive with the environment.yml that is in R:\River Delta Land\Metronome\python
            - Copy the .yml file to user drive
            - Command in anaconda prompt = conda env create -f environment.yml
    - Activate the Metronome environment
    - Start Spyder by using the command 'spyder'
    
Note: IF YOU ENCOUNTER AN ERROR WITH 'FPS' -> CHECK THAT YOU HAVE THE FOLLOWING VERSIONS OF THE PACKAGES INSTALLED:
    - imageio==2.19.3
    - imageio-ffmpeg==0.4.7  

Author: Eise W. Nota (finalized AUGUST 2024)    
'''

#%% Clearing all variables
from IPython import get_ipython
get_ipython().magic('reset -sf') 

#%% Import packages
import os
import shutil
import pandas as pd
import datetime
from datetime import timedelta

# Set base directory of the network drive
pwd =  r"\\..."

# Import functions
os.chdir(pwd + r"\python\Overhead_scripts\Functions_Overhead") # Setting the functions directory
from Function_create_timelapse_overhead import create_metronome_movie_Metashape
os.chdir(pwd + r"\python\Overhead_scripts\Functions_Overhead") 
from Function_flatten_list import flatten_list
os.chdir(pwd + r"\python\Overhead_scripts\Functions_Overhead") 

start = datetime.datetime.now()

#%% Set the baseModel you want to use
baseModel = '11'

#%% Defining the experiment-specific variables
Expnr = '054'                                                                   # Experiment number
pilot = 1                                                                       # Pilot number, 0 if no pilot
start_cycle = '00000'                                                           # String of cycle where you want to start # 00000 = first cycle
end_cycle = '01000'                                                             # String of cycle where you want to end # End with complete numbers as in folder structure 
nr_cycles = int(end_cycle) - int(start_cycle)                                   # Integer of total amount of cycles  
cycledurat = 40                                                                 # Seconds of one tidal cycle
orthoRes = 5                                                                    # Desired image resolution in mm; 5 is the default for timelapse
tilting_imagery=False                                                           # False if Metronome = horizontal, else True (Metronome imagery during tilting)
startXY = (0,0)                                                                 # At what x-coordinates do we want the orthomosiac to start? # Default (0,0)
endXY = (20,3)                                                                  # At what x-coordinates do we want the orthomosiac to end? # Default (20,3)

# Movie parameters
averaged_over = 1                                                               # Amount of images used for 1 frame in the movie
calculateMean = 1                                                               # Frame used for mean of of the averaged_over cycles # 1 yes, 0 no
calculateMedian = 0                                                             # Frame used for median of of the averaged_over cycles # 1 yes, 0 no
jump = 1                                                                       # What is the interval of images you want to use for making the movie? #20      
fps = 15                                                                        # Frames per second
target_image_size = (5600,896)                                                  # Adjust the target size as needed for correct saving (11194,1776) if not downsized
figWidth = 30                                                                   # Width in matplotlib   
figHeight = 5     

# Do we need to skip folders? E.g. PIV / empty folders / 1 Hz measurements
skipFolders = 0
# If so, which folders? # They are stored in R:\Metronome\python\Calibration_scripts_and_parameters\CalibrationParameters
if skipFolders == 1:
    foldersToSkip = ['']                                             # List of all folders to skip, e.g. ['06292-08000'] 
    # Update nr_cycles
    for folder in foldersToSkip:
        # How many actual cycles are skipped?
        nr_cycles = nr_cycles - (int(folder[6:])-int(folder[:5]))
    del(folder)
else: # define foldersToSkip as empty list (which you don't need but require for movie function input)
    foldersToSkip = []

# Image and camera parameters
cams = [1,2,3,4,5,6,7]                                                          # Defining number of overhead cameras

# Choosing colours for images 
rgb_img = 1                                                                     # 1 is true, 0 is false

#%% Set RGB or LAB settings
if rgb_img == 1:
    form = '_RGB_'
    clrmapint  = 'none' # not used, but required for movie function
else:
    raise ValueError('No RGB condition set.')

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
    writeFolder = pwd + r'\experiments\Exp' + Expnr + '\\processed_data\\timelapse_videos'
    orthoFolder = pwd + r'\experiments\Exp' + Expnr + '\\processed_data\\orthophotos\\overhead_cameras\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\'
    namestart = 'Exp' + Expnr + '_BaseModel' + baseModel + form
else:
    rootFolder = pwd + r'\experiments\Exp' + Expnr + '\Pilot' + str(pilot) + '\\raw_data\\overhead_cameras'
    copyFolder = pwd + r'\experiments\Exp' + Expnr + '\Pilot' + str(pilot) + '\\processed_data\\copyBaseModel'
    copyRawFolder = pwd + r'\experiments\Exp' + Expnr + '\Pilot' + str(pilot) + '\\processed_data\\Raw_photos'
    writeFolder = pwd + r'\experiments\Exp' + Expnr + '\Pilot' + str(pilot) + '\\processed_data\\timelapse_videos'
    orthoFolder = pwd + r'\experiments\Exp' + Expnr + '\Pilot' + str(pilot) + '\\processed_data\\orthophotos\\overhead_cameras\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\'
    namestart = 'Exp' + Expnr + '_Pilot' + str(pilot) + '_BaseModel' + baseModel + form
    
# Create a directory if it does not exist
if not os.path.exists(orthoFolder):
    os.makedirs(orthoFolder)    
    
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
    
# Defining variables create_movie function
output_path = writeFolder + '\\' + namestart + start_cycle + '-' + end_cycle + '_overhead_orthomosaics_avg' + str(averaged_over) + '_int' + str(jump) + '_fps' + str(fps) + '.mp4' # .avi if use codec XVID

#%% Load calibration parameters
# Computing a correct orthomosaic requires some (pre-)processing steps, which have been determined in previous calibration analyses 
# These calibration parameters have been determined in either Python codes or the Agisoft BaseModel
# We load these files in the main code, instead of in the separate functions, to enhance the understandibility of the codes

# All of the overhead cameras have their own color deviations, so initial white balancing based on the DSLR survey in the BaseModel is required
initialWBCSV = pd.read_csv(calibrationFolder + r"\\Initial_white_balancing.csv",delimiter=',',header=None) # Calibrated with CalibrateOverheadCameraColors.py

# Load marker coordinates of the Overheads, as determined in the Agisoft BaseModel # Only necessary if tilting_imagery = False
if tilting_imagery == False:
    markerCSV = pd.read_csv(calibrationFolder + r"\\Overhead_markers_positions.csv",delimiter=',')
else:
    markerCSV = False

#%% Loading input directories
amount_folders = len(os.listdir(rootFolder + '\cam_1')) # The amount of folders should be the same for all cameras
chosen_folders = {}
      
# Create dictionary of available folders per camera while selecting only folders based on start_cycle and end_cycle
for i in cams:
    available_folders = os.listdir(rootFolder + '\cam_' + str(i))
    pick_folders = []
    for j in available_folders:
        if 'Thumbs' not in j:
            if int(start_cycle) <= int(j[:5]) < int(end_cycle): # Use < end_cycle else it includes the next folder
                pick_folders.append(j)
    # If we want to exclude folders, test if the excluded folders are in the pick_folders list
    if skipFolders == 1:
        pick_folders = [x for x in pick_folders if x not in foldersToSkip] # Remove all foldersToSkip
    # Add pick_folders to chosen_folders        
    chosen_folders['\cam_{0}'.format(i)] = pick_folders
    
del(available_folders)

#%% Create dictionary of available filenames per camera
file_names = {}  

for key in chosen_folders:
    cam_name = key # e.g. '\\cam_1'
    cam_num = int(cam_name[-1])
    folders = chosen_folders[key]
    
    # Looping through the folders to fill cycle and filenames in dictionaries
    for f in range(len(folders)):
        # Creating a list of which cycles are available in the folder
        cycles = []
        avail_cyclenames = os.listdir(rootFolder +'\cam_' + str(cam_num) + '\\' + folders[f])
        cycles.append([file for file in avail_cyclenames if file.endswith('.bmp')])
        cycles_flat = flatten_list(cycles)
        # Add the cycles to dictionaries
        for c in range(len(cycles_flat)):
            #cycle_names.setdefault(str(cam_name),[]).append(cycles_flat[c])
            file_names.setdefault(str(cam_name),[]).append(rootFolder + cam_name + '\\' + folders[f] + '\\' + cycles_flat[c])
        
    del(avail_cyclenames, cycles_flat, folders, cam_name, cam_num)

# Test whether all elements in the folder have the same length (i.e. each camera has the same amount of photos)
lens = map(len, file_names.values())
if len(set(lens)) == 0:
    raise ValueError('Not all cameras have the same amount of files')
    
del(lens)
        
# We now have a list of all files in the relevant folders, but that doesn't mean that these represent all the files desired for creating the video
# if we don't want to use the final folder completely
# The assumption is here that we always start with a cycle that includes the beginning of a folder and not somewhere later
if nr_cycles != len(file_names.get('\cam_1')):
    # There is a possibility that nr_cycles is larger than len(file_names), in that case raise exception
    if nr_cycles > len(file_names.get('\cam_1')):
        raise Exception('Chosen folders contain less files than their names indicate. Manually update folder names.')
    else: # nr_cycles is less
        # The residual amount of cycles can be deleted from the dictionaries
        cycles_to_delete = len(file_names.get('\cam_1'))-nr_cycles
        # Delete the undesired files from the dictionary
        for key in file_names:
            del(file_names[key][-cycles_to_delete:])
    
#%% Now determine which of these files we need to use based on jump interval and averaged_over
rel_files = {}        
for key in file_names:
    for k in range(0,nr_cycles,jump): # Consider the jump first
        m_end = k + averaged_over - 1 # test the final number of m's
        for l in range(averaged_over): # Now consider the averaging of subsequent cycles
            m = k + l # Add both k and l for proper index used
            # Can only be run if the final cycle is within range of total averaged_over
            if m_end < nr_cycles:
                rel_files.setdefault(key,[]).append(file_names.setdefault(key)[m])
                
del(key,file_names,k,m,m_end,l)

#%% Now we have all the files, create the movie from Function_create_movie_overhead_labelled file
create_metronome_movie_Metashape(rel_files,output_path,fps,averaged_over,nr_cycles,start_cycle,jump,cycledurat,form,
                                     Expnr,pilot,figWidth,figHeight,target_image_size,initialWBCSV,skipFolders,foldersToSkip,baseModel,
                                     orthoFolder,copyFolder,calculateMean,calculateMedian,orthoRes,startXY,endXY,markerCSV,tilting_imagery)

# How long did the run take?
end = datetime.datetime.now()
td = round((end - start).total_seconds())
print("Script running time = " + str(timedelta(seconds=td)))