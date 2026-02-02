'''
Code to create water depth maps from overhead orthomosaics, including timelapse videos from
Metronome in the Earth Simulation Laboratory

The orthomosaics need to be computed through base model alignment (Overhead_Orthomosaics_Metashape_Timelapse.py)
A mask needs to be implemented as well (DEM2Mask4Ortho_DepthMaps.py)

Note: this code works for joblib and sklearn version 1.3.2

Note: if dye concentration is unknown, model fitting needs to be conducted in Overhead_testDyeConcentrations4WaterDepthMaps.py.

Note: the depth maps are calculated through machine learning (random forest) models, which are base model-specific.
All the model development and validation can be found in \\RandomForestModels

Note: this code works both for processing timeseries (option fixedCycle) and tidal cyles imagery (option 1Hz)

Note: the Metronome has cartesian coordinates: 
x = 0-20 m along the flume to the right
y = 0-3 m across the flume to the back wall
z = 0-0.4 m elevation above the flume floor

Note: To run this script you need to install packages from the environment.yml:
    - Create the Metronome environment on your user drive with the environment.yml that can be found on GitHub https://github.com/UtrechtRiversEstuaries/MetronomeDataProcessingPython:
            - Copy the .yml file to user drive
            - Command in anaconda prompt = conda env create -f environment.yml
    - Activate the Metronome environment
    - Start Spyder by using the command 'spyder'
    
Note: IF YOU ENCOUNTER AN ERROR WITH 'FPS' -> CHECK THAT YOU HAVE THE FOLLOWING VERSIONS OF THE PACKAGES INSTALLED:
    - imageio==2.19.3
    - imageio-ffmpeg==0.4.7  

Author: Eise W. Nota (finalized SEPTEMBER 2025)    

'''

#%% Clearing all variables
from IPython import get_ipython
get_ipython().magic('reset -sf') 

#%% Import packages
import os
import datetime
from datetime import timedelta

# Set base directory of the network drive
pwd =  r"\\..."

# Set the directory of the functions
os.chdir(pwd + r"\python\Overhead_scripts\Functions_Overhead") # Setting the functions directory
from Function_create_timelapse_overhead_depthMaps import create_metronome_movie_depthMaps

start = datetime.datetime.now()

#%% Defining the experiment-specific variables
Expnr = '062'                                                                   # Experiment number
pilot = 0                                                                       # Pilot number, 0 if no pilot
start_cycle = '00000'                                                           # String of orthomosaic cycle where you want to start # 00000 = first cycle
end_cycle = '09000'                                                             # String of orthomosaic cycle where you want to end # End with complete numbers as in folder structure 
nr_cycles = int(end_cycle) - int(start_cycle)                                   # Integer of total amount of cycles  
cycledurat = 40                                                                 # Seconds of one tidal cycle
orthoRes = 25                                                                   # Desired image resolution in mm; 5 is the default for timelapse
startXY = (0,0)                                                                 # At what x-coordinates do we want the orthomosiac to start? # Default (0,0)
endXY = (20,3)   
# In case you want a zoomed-in section to be shown, change below
startXY2 = (0,0)                                                                # Default (0,0)
endXY2 = (20,3)       
imagery2beProcessed = 'fixedCycle'                                              # 'fixedCycle'; '1Hz'

# Dye concentration
concentration = '5' # Default is '2' for recent experiments; options are '0'; '1'; '2'; '3'; '4'; '5'

# What kind of video do you want to make?
# Options: 'waterDepth'; 'waterDepthElevation'; 'combined'
videoStyle = 'waterDepth'

# The normal code only makes a video of depth maps/elevation
# If you want a fancy video which includes orthomosaics and classifier dry cells with depth maps, set below to 1
fancyVideo = 1

if fancyVideo == 1:
    # What is the resolution of illustrative orthos? 5 mm defaulft
    illOrthoRes = 5
else:
    if videoStyle == 'combined':
        raise Exception('If videoStyle is combined, fancyVideo has to be 1')

# Do we want a fancy video combining orthos

# What is the mask information?
maskExpnr = '062'
maskCycle = '00000'
maskBuffer = '1'
maskpilot = 0

# Movie parameters
averaged_over = 1  # Default 1
jump = 20                                                                       # What is the interval of images you want to use for making the movie? #20      
fps = 5                                                                        # Frames per second

# Do we need to skip cycle? I.e. if the first timestep after surveys are included and there is no hydrostatic equilibrium
skipCycles = 0
# If so, which cycles? # They are stored in R:\Metronome\python\Calibration_scripts_and_parameters\CalibrationParameters
if skipCycles == 1:
    cyclesToSkip = ['']
else: # define cyclesToSkip as empty list (which you don't need but require for movie function input)
    cyclesToSkip = []

# Image size-dependent
if startXY2 == startXY and endXY == endXY2: # Default
    if fancyVideo == 1:
        if videoStyle == 'combined':
            target_image_size = (5600,3000)                                              # Adjust the target size as needed for correct saving (11194,1776) if not downsized
            figWidth = 30                                                               # Width in matplotlib   
            figHeight = 20  
        else:
            target_image_size = (5600,2100)                                              # Adjust the target size as needed for correct saving (11194,1776) if not downsized
            figWidth = 30                                                               # Width in matplotlib   
            figHeight = 13  
    else:
        target_image_size = (5600,896)                                              # Adjust the target size as needed for correct saving (11194,1776) if not downsized
        figWidth = 30                                                               # Width in matplotlib   
        figHeight = 5                                                               # Height in matplotlib
else: # Likely manual adjustment
    target_image_size = (5600,1362)                                              # Adjust the target size as needed for correct saving (11194,1776) if not downsized
    figWidth = 30                                                               # Width in matplotlib   
    figHeight = 7  

#%% Some default settings, which are equal for all ML models
baseModel = '11'
# Weir parameters
finalWeirLevel = 0.08613
demfloodLevel = 0.102
# All other can be found in \\soliscom.uu.nl\GEO\FG\River Delta Land\SandyNetworksMetronome_Eise_Nota\Blueness

#%% Defining the root- and writefolder based on input variables
modelFolder = pwd + r"\python\ML_Overhead_water_depth_models\BaseModel" + baseModel + "\\Res" + str(orthoRes) + "mm\\"

# Defining the root- and writefolders based on the experiment and setting
if pilot == 0:
    maskFolder =  pwd + r'\experiments\Exp' + maskExpnr + '\\derived_data\\masks\\laser_scanner\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\'
    movieFolder = pwd + r'\experiments\Exp' + Expnr + '\\processed_data\\timelapse_videos'
    if imagery2beProcessed == 'fixedCycle':
        orthoFolder = pwd + r'\experiments\Exp' + Expnr + '\\processed_data\\orthophotos\\overhead_cameras\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\'
        writeFolder = pwd + r'\experiments\Exp' + Expnr + '\\derived_data\\overhead_WaterDepthMaps\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\Conc' + concentration + '\\'        
        if fancyVideo == 1:
            illOrthoFolder = pwd + r'\experiments\Exp' + Expnr + '\\processed_data\\orthophotos\\overhead_cameras\\BaseModel' + baseModel + '\\Res' + str(illOrthoRes) + 'mm\\'
    elif imagery2beProcessed == '1Hz':
        orthoFolder = pwd + r'\experiments\Exp' + Expnr + '\\processed_data\\orthophotos\\overhead_cameras_1Hz\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\'
        writeFolder = pwd + r'\experiments\Exp' + Expnr + '\\derived_data\\overhead_WaterDepthMaps_1Hz\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\Conc' + concentration + '\\'
        if fancyVideo == 1:
            illOrthoFolder = pwd + r'\experiments\Exp' + Expnr + '\\processed_data\\orthophotos\\overhead_cameras_1Hz\\BaseModel' + baseModel + '\\Res' + str(illOrthoRes) + 'mm\\'
    if videoStyle == 'waterDepth':
        namestart = 'Exp' + Expnr + '_WaterDepthMaps_BaseModel' + baseModel + '_Res' + str(orthoRes) + 'mm_Conc' + concentration + '_'
    elif videoStyle == 'waterDepthElevation':
        namestart = 'Exp' + Expnr + '_WaterDepthDEMMaps_BaseModel' + baseModel + '_Res' + str(orthoRes) + 'mm_Conc' + concentration + '_'
    elif videoStyle == 'combined':
        namestart = 'Exp' + Expnr + '_WaterDepthCombined_BaseModel' + baseModel + '_Res' + str(orthoRes) + 'mm_Conc' + concentration + '_'
else:
    maskFolder =  pwd + r'\experiments\Exp' + maskExpnr + '\Pilot' + str(pilot) + '\\derived_data\\masks\\laser_scanner\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\'
    movieFolder = pwd + r'\experiments\Exp' + Expnr + '\Pilot' + str(pilot) + '\\processed_data\\timelapse_videos'
    if imagery2beProcessed == 'fixedCycle':
        orthoFolder = pwd + r'\experiments\Exp' + Expnr + '\Pilot' + str(pilot) + '\\processed_data\\orthophotos\\overhead_cameras\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\'
        writeFolder = pwd + r'\experiments\Exp' + Expnr + '\Pilot' + str(pilot) + '\\derived_data\\overhead_WaterDepthMaps\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\Conc' + concentration + '\\'
        if fancyVideo == 1:
            illOrthoFolder = pwd + r'\experiments\Exp' + Expnr + '\Pilot' + str(pilot) + '\\processed_data\\orthophotos\\overhead_cameras\\BaseModel' + baseModel + '\\Res' + str(illOrthoRes) + 'mm\\'
    elif imagery2beProcessed == '1Hz':
        orthoFolder = pwd + r'\experiments\Exp' + Expnr + '\Pilot' + str(pilot) + '\\processed_data\\orthophotos\\overhead_cameras_1Hz\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\'
        writeFolder = pwd + r'\experiments\Exp' + Expnr + '\Pilot' + str(pilot) + '\\derived_data\\overhead_WaterDepthMaps_1Hz\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\Conc' + concentration + '\\'
        if fancyVideo == 1:
            illOrthoFolder = pwd + r'\experiments\Exp' + Expnr + '\Pilot' + str(pilot) + '\\processed_data\\orthophotos\\overhead_cameras_1Hz\\BaseModel' + baseModel + '\\Res' + str(illOrthoRes) + 'mm\\'
    if videoStyle == 'waterDepth':
        namestart = 'Exp' + Expnr + '_Pilot' + str(pilot) + '_WaterDepthMaps_BaseModel' + baseModel + '_Res' + str(orthoRes) + 'mm_Conc' + concentration + '_'
    elif videoStyle == 'waterDepthElevation':
        namestart = 'Exp' + Expnr + '_Pilot' + str(pilot) + '_WaterDepthDEMMaps_BaseModel' + baseModel + '_Res' + str(orthoRes) + 'mm_Conc' + concentration + '_'
    elif videoStyle == 'combined':
        namestart = 'Exp' + Expnr + '_Pilot' + str(pilot) + '_WaterDepthCombined_BaseModel' + baseModel + '_Res' + str(orthoRes) + 'mm_Conc' + concentration + '_'

if fancyVideo == 1:
    namestart = namestart + 'fancy_'

if 'illOrthoFolder' not in locals():
    illOrthoFolder = False
    
# Defining variables create_movie function
# Depending on whether zoomed in or not
if startXY2 == startXY and endXY == endXY2:
    if imagery2beProcessed == 'fixedCycle':
        output_path = movieFolder + '\\' + namestart + start_cycle + '-' + end_cycle + '_avg' + str(averaged_over) + '_int' + str(jump) + '_fps' + str(fps) + '.mp4' # .avi if use codec XVID
    elif imagery2beProcessed == '1Hz':
        output_path = movieFolder + '\\' + namestart + start_cycle + '-' + end_cycle + '_1Hz_fps' + str(fps) + '.mp4' # .avi if use codec XVID
else:
    if imagery2beProcessed == 'fixedCycle':
        output_path = movieFolder + '\\' + namestart + start_cycle + '-' + end_cycle + '_avg' + str(averaged_over) + '_int' + str(jump) + '_fps' + str(fps) + '_zoomed.mp4' # .avi if use codec XVID
    elif imagery2beProcessed == '1Hz':
        output_path = movieFolder + '\\' + namestart + start_cycle + '-' + end_cycle + '_1Hz_fps' + str(fps) + '_zoomed.mp4' # .avi if use codec XVID

# Define relevant files
# Mask and to writeFolder
if maskpilot == 0:
    if int(maskBuffer) > 0:
        maskDir = maskFolder + 'MaskExp' + maskExpnr + 'Cycle' + maskCycle + 'Buffer' + maskBuffer + '.nc'
        writeFolder = writeFolder + 'Mask' + maskExpnr + 'Cycle' + maskCycle + 'Buffer' + maskBuffer + '\\'
    else:
        maskDir = maskFolder + 'MaskExp' + maskExpnr + 'Cycle' + maskCycle + '.nc'
        writeFolder = writeFolder + 'Mask' + maskExpnr + 'Cycle' + maskCycle + '\\'
else:
    if int(maskBuffer) > 0:
        maskDir = maskFolder + 'MaskExp' + maskExpnr + 'Pilot' + str(maskpilot) + 'Cycle' + maskCycle + 'Buffer' + maskBuffer + '.nc'
        writeFolder = writeFolder + 'Mask' + maskExpnr + 'Pilot' + str(maskpilot) + 'Cycle' + maskCycle + 'Buffer' + maskBuffer + '\\'
    else:
        maskDir = maskFolder + 'MaskExp' + maskExpnr + 'Pilot' + str(maskpilot) + 'Cycle' + maskCycle + '.nc'
        writeFolder = writeFolder + 'Mask' + maskExpnr + 'Pilot' + str(maskpilot) + 'Cycle' + maskCycle + '\\'
    
# Create a directory if it does not exist
if not os.path.exists(writeFolder):
    os.makedirs(writeFolder)        
    
# Models
clfDir = modelFolder + 'Conc' + concentration + 'Clf.joblib'
regrDir = modelFolder + 'Conc' + concentration + 'Regr.joblib'

#%% Create list of available orthomosaics
avail_files = os.listdir(orthoFolder)
# Determine relevant orthos
rel_orthos = []
for i in avail_files:
    if imagery2beProcessed == 'fixedCycle':
        if 'Thumbs' not in i:
            if pilot == 0:
                if i[11:16] not in cyclesToSkip:
                    if int(start_cycle) <= int(i[11:16]) <= int(end_cycle): 
                        rel_orthos.append(i)
            else:
                if i[18:23] not in cyclesToSkip:
                    if int(start_cycle) <= int(i[18:23]) <= int(end_cycle): 
                        rel_orthos.append(i)
    elif imagery2beProcessed == '1Hz':
        if 'Thumbs' not in i:
            if pilot == 0:
                if i[23:28] not in cyclesToSkip:
                    if int(start_cycle) <= int(i[23:28]) < int(end_cycle): # Use < end_cycle else it includes the next folder
                        rel_orthos.append(i)
            else:
                if i[30:35] not in cyclesToSkip:
                    if int(start_cycle) <= int(i[30:35]) < int(end_cycle): # Use < end_cycle else it includes the next folder
                        rel_orthos.append(i)

# Apparently it's necessary for some experiments to sort
if imagery2beProcessed == 'fixedCycle':
    rel_orthos = sorted(rel_orthos, key=lambda x: x.split('_')[2])
                    
#%% Now we have all the files, create the movie from Function_create_movie_depth_maps file
create_metronome_movie_depthMaps(rel_orthos,nr_cycles,start_cycle,cycledurat,Expnr,pilot,figWidth,figHeight,
                                 target_image_size,writeFolder,orthoFolder,orthoRes,output_path,startXY,endXY,
                                 startXY2,endXY2,fps,maskDir,clfDir,regrDir,videoStyle,finalWeirLevel,demfloodLevel,
                                 imagery2beProcessed,fancyVideo,illOrthoFolder)
    
#%% How long did the run take?
end = datetime.datetime.now()
td = round((end - start).total_seconds())
print("Script running time = " + str(timedelta(seconds=td)))    
