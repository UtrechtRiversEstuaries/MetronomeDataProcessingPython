'''
Code to create overhead orthomosaics and from the overhead cameras of the 
Metronome blueness final calibration dataset to RGB to another colorspace
The frames are created through generating orthomosaics from the overhead 
imagery using Agisoft Metashape

Python script written by Eise W. Nota (finalized JANUARY 2026)
    
'''

#%% Clearing all variables
from IPython import get_ipython
get_ipython().magic('reset -sf') 

#%% Import packages
import cv2
import os
import datetime
from datetime import timedelta

# Set base directory of the network drive
pwd =  r"..."

from Function_colour_correction_overhead import rgb_to_LAB, rgb_to_Luv, rgb_to_YCrCb, rgb_to_HSV

start = datetime.datetime.now()

#%% Defining the variables
Expnr = '058'                                   # Experiment number
baseModel = '11'
orthoRes = 25 # mm
colorspaces = ['LAB','Luv','YCrCb','HSV']  
# Gaussian smooth settings
smGsettings = [5,5,0,0]

# Determine root and write directories
rootFolder = pwd +  r'\\smG\\RGB\\'
writefolder = pwd +  r'\\smG\\'
    
# Create folder if the directory does not exist
if not os.path.exists(writefolder):
    os.makedirs(writefolder)

#%% Loop over the colorspaces
for colorspace in colorspaces:        
    #% List all the relevant concentrations folders  
    rgb_orthos_folders = os.listdir(rootFolder)
    
    # Now loop through the folders
    for concentration in rgb_orthos_folders:
        # Update root folder
        currentRootFolder = rootFolder + concentration
        
        # Now list all relevant orthos
        currentOrthos = os.listdir(currentRootFolder)
        # Loop through these orthos
        for ortho in currentOrthos:
            # Filter out thumbs files
            if 'Thumbs' not in ortho:
                # What is the writefile?
                writeFile = ortho.replace('RGB',colorspace)
                # Load the RGB ortho
                currentOrtho = cv2.imread(currentRootFolder + '\\' + ortho)
                # Convert to colorspace and
                if colorspace == 'LAB':
                    converseOrtho = rgb_to_LAB(currentOrtho)
                elif colorspace == 'Luv':
                    converseOrtho = rgb_to_Luv(currentOrtho)
                elif colorspace == 'YCrCb':
                    converseOrtho = rgb_to_YCrCb(currentOrtho)
                elif colorspace == 'HSV':
                    converseOrtho = rgb_to_HSV(currentOrtho)
                
                # Update writeFolder
                currentWriteFolder = writefolder + colorspace + '\\' + concentration
            
                # Create folder if the directory does not exist
                if not os.path.exists(currentWriteFolder):
                    os.makedirs(currentWriteFolder)
        
                # And store
                cv2.imwrite(currentWriteFolder + '\\' + writeFile, converseOrtho)
                print(writeFile + ' stored as ' + colorspace)
            
#%% How long did the run take?
end = datetime.datetime.now()
td = round((end - start).total_seconds())
print("Script running time = " + str(timedelta(seconds=td)))