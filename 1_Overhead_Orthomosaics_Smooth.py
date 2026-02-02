'''
Code to denoise RGB overhead orthomosaics from the Metronome 
in the Earth Simulation Laboratory using Gaussian smoothening

Python script written by Eise W. Nota (finalized JANUARY 2026)
    
'''

#%% Clearing all variables
from IPython import get_ipython
get_ipython().magic('reset -sf') 


#%% Import packages
import cv2
import os
import pandas as pd
import numpy as np
import datetime
from datetime import timedelta

# Set base directory of the network drive
pwd =  r"..."

start = datetime.datetime.now()

#%% Defining the variables
Expnr = '058'                                   # Experiment number
baseModel = '11'
orthoRes = 25 # mm
applyMask = True # If false, the wall boundary may have too much of an effect on the smoothening/denoising
borderType = cv2.BORDER_ISOLATED # Default borderType (tested in previous script)
# Gaussian smooth settings
smGsettings = [5,5,0,0]

#%% Determine root and write directories
rootFolder = pwd + r'\\rootFolder\\'
writefolder = pwd +  r'\\smG\\RGB\\'
    
# Create folder if the directory does not exist
if not os.path.exists(writefolder):
    os.makedirs(writefolder)
    
#%% If relevant, load mask
if applyMask == True:
    whichMask = '2' # default
    maskFolder = pwd + r'\\mask'
    # Load mask
    pdMask = pd.read_csv(maskFolder + '\\mask' + whichMask + '.csv', delimiter=',', header=None)     
    # Make array
    mask = np.array(pdMask)
    # Set zeros to NaN
    mask = mask.astype('float')
    mask[mask == 0] = np.nan

#%% List all the relevant concentrations folders  
rgb_orthos_folders = os.listdir(rootFolder)

# Now loop through the folders
for concentration in rgb_orthos_folders:
    # Update root- and writefolders
    currentRootFolder = rootFolder + concentration
    currentWriteFolder = writefolder + concentration
    # Create folder if the directory does not exist
    if not os.path.exists(currentWriteFolder):
        os.makedirs(currentWriteFolder)
    
    # Now list all relevant orthos
    currentOrthos = os.listdir(currentRootFolder)
    # Loop through these orthos
    for ortho in currentOrthos:
        # Filter out thumbs files
        if 'Thumbs' not in ortho:
            # Load the RGB ortho
            currentOrtho = cv2.imread(currentRootFolder + '\\' + ortho)
            # If set, apply mask
            if applyMask == True:
                orthoMask0 = np.zeros([np.size(currentOrtho,0),np.size(currentOrtho,1),3])
                orthoMask0[:,:,0] = mask; orthoMask0[:,:,1] = mask; orthoMask0[:,:,2] = mask
                currentOrtho = np.array(currentOrtho*orthoMask0,dtype='uint8')
                # Delete rows and columns with all zeros
                currentOrtho = currentOrtho[~(currentOrtho == 0).all(axis=(1,2))]
                currentOrtho = currentOrtho[:,~(currentOrtho == 0).all(axis=(0,2))]
            # Apply Gaussian smoothening
            # What is the writefile?
            writeFile = ortho.replace('RGB','RGB_smG')
            dst = cv2.GaussianBlur(currentOrtho,(smGsettings[0],smGsettings[1]),smGsettings[2],smGsettings[3],borderType = borderType)
    
            cv2.imwrite(currentWriteFolder + '\\' + writeFile, dst)
            print(writeFile + ' stored')
            
#%% How long did the run take?
end = datetime.datetime.now()
td = round((end - start).total_seconds())
print("Script running time = " + str(timedelta(seconds=td)))