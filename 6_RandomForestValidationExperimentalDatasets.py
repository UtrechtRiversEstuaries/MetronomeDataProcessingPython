'''
Code to apply validation of the random forest models and extract all the relevant information.
The validations in this script are applicable for the experimental datasets.

Python script written by Eise W. Nota (finalized JANUARY 2026)

'''

#%% Clearing all variables
from IPython import get_ipython
get_ipython().magic('reset -sf') 

#%% Import packages
import cv2
import os
import numpy as np
import netCDF4 as nc
import datetime
import csv
import matplotlib.pyplot as plt
from sklearn import metrics
import joblib
import scores.categorical
import xarray as xr

# Set base directory of the network drive
pwd =  r"..."

from Function_colour_correction_overhead import rgb_to_LAB, rgb_to_Luv, rgb_to_YCrCb, rgb_to_HSV

start = datetime.datetime.now()

#%% Defining the variables
baseModel = '11'
orthoRes = 25

# Relevant orthos
Exp062_01000_6900 = 'Exp058_Exp062_01000_weir6900'
Exp062_03000_6900 = 'Exp058_Exp062_03000_weir6900'
Exp062_09000_4636 = 'Exp058_Exp062_09000_weir4636'
Exp062_09000_7570 = 'Exp058_Exp062_09000_weir7570'
Exp062_09000_8327 = 'Exp058_Exp062_09000_weir8327'
Exp077_12675_6700 = 'Exp077_Exp077_12675_weir6700'

Exp062_orthos = [Exp062_01000_6900,Exp062_03000_6900,Exp062_09000_4636,Exp062_09000_7570,Exp062_09000_8327]
Exp062_dems = ['01000','03000','09000','09000','09000']
Exp062_weirs = ['6900','6900','4636','7570','8327']
Exp077_dem = '12675'
Exp077_weir = '6700'

# Relevant masks
mask062 = '062'
maskCycle062 = '00000'
maskBuffer062 = '2'
mask077 = '078'
maskCycle077 = '00000'
maskBuffer077 = '1'

# Pre-processing method for ortho
borderTypeO = cv2.BORDER_ISOLATED # Default borderType
smGsettings = [5,5,0,0]

# Weir parameters
finalWeirLevel = 0.08613
differenceWaterLevel = 0.00 # Maybe 0 because of capillary forces
demfloodLevel = 0.102
weirOffset = round((demfloodLevel - differenceWaterLevel) - finalWeirLevel,5)

# Models to test
concentrations = ['0','1','2','3','4','5']

#%% Determine relevant directories and files
prepFolder = pwd + r'\\SandyNetworksMetronome_Eise_Nota\\Blueness\\Figures\\Fig_3_2_1_prep\\'
demFolder062 = pwd + r'\\Metronome\\experiments\\Exp062\\processed_data\\DEMs\\laser_scanner\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\'
demFolder077 = pwd + r'\\Metronome\\experiments\\Exp077\\processed_data\\DEMs\\laser_scanner\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\'
maskFolder062 = pwd + r'\\Metronome\\experiments\\Exp' + mask062 + '\\derived_data\\masks\\laser_scanner\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\'
maskFolder077 = pwd + r'\\Metronome\\experiments\\\Exp' + mask077 + '\\derived_data\\masks\\laser_scanner\\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\'
modelFolder = pwd + r'\\Metronome\python\ML_Overhead_water_depth_models\BaseModel' + baseModel + '\Res' + str(orthoRes) + 'mm'

# Load masks
if int(maskBuffer062) > 0:
    maskFile062 = 'MaskExp' + mask062 + 'Cycle' + maskCycle062 + 'Buffer' + maskBuffer062 + '.nc'
else:
    maskFile062 = 'MaskExp' + mask062 + 'Cycle' + maskCycle062 + '.nc'
    
if int(maskBuffer077) > 0:
    maskFile077 = 'MaskExp' + mask077 + 'Cycle' + maskCycle077 + 'Buffer' + maskBuffer077 + '.nc'
else:
    maskFile077 = 'MaskExp' + mask077 + 'Cycle' + maskCycle077 + '.nc'

# DEM files
demFile077 = 'Exp077_' + Exp077_dem + '_Laserscan_DEM.nc'
demFiles062 = []
for i in range(len(Exp062_dems)):
    demFiles062.append('Exp062_' + Exp062_dems[i] + '_Laserscan_DEM.nc')

#%% Load masks
# Exp062
maskNC062 = nc.Dataset(maskFolder062 + '\\' + maskFile062, 'r')
# Extract the relevant information
mask062 = np.array(maskNC062.get_variables_by_attributes(name='mask')[0])
# Extract the relevant information
xAxis = np.array(maskNC062.get_variables_by_attributes(name='X-axis')[0])
yAxis = np.array(maskNC062.get_variables_by_attributes(name='Y-axis')[0])
# We can close the file again
maskNC062.close()
# Flip vertically for ortho
mask062 = np.flipud(mask062)

# Convert the X/Y-axis information to plottable values
xValues = np.arange(xAxis[0],xAxis[1],xAxis[2])
yValues = np.arange(yAxis[0],yAxis[1],yAxis[2])
xx, yy = np.meshgrid(xValues,yValues)

# Also create a stacked mask
maskStacked062 = mask062.reshape(-1)
boolMaskStacked062 = maskStacked062.astype(bool)

# Make mask suited for 3D
orthoMask3062 = np.zeros([np.size(mask062,0),np.size(mask062,1),3])
orthoMask3062[:,:,0] = mask062; orthoMask3062[:,:,1] = mask062; orthoMask3062[:,:,2] = mask062

# Exp077
maskNC077 = nc.Dataset(maskFolder077 + '\\' + maskFile077, 'r')
# Extract the relevant information
mask077 = np.array(maskNC077.get_variables_by_attributes(name='mask')[0])
# We can close the file again
maskNC077.close()
# Flip vertically for ortho
mask077 = np.flipud(mask077)

# Also create a stacked mask
maskStacked077 = mask077.reshape(-1)
boolMaskStacked077 = maskStacked077.astype(bool)

# Make mask suited for 3D
orthoMask3077 = np.zeros([np.size(mask077,0),np.size(mask077,1),3])
orthoMask3077[:,:,0] = mask077; orthoMask3077[:,:,1] = mask077; orthoMask3077[:,:,2] = mask077

# Define additional mask for Exp062_09000_4636 due to the upstream section not being fully drained when imagery was taken
mask062_4636 = mask062.copy()
mask062_4636[:,:260] = 0
mask062_4636[70:,:400] = 0
mask062_4636[80:,:520] = 0
mask062_4636[90:,:560] = 0

# Make mask suited for 3D
orthoMask3062_4636 = np.zeros([np.size(mask062_4636,0),np.size(mask062_4636,1),3])
orthoMask3062_4636[:,:,0] = mask062_4636; orthoMask3062_4636[:,:,1] = mask062_4636; orthoMask3062_4636[:,:,2] = mask062_4636

#%% Plot to see whether this is the appropriate mask
figwidth = 30 # inches
figsize=(figwidth, figwidth*(3/20)) # scale y- and x-axes
labelsize = 15 # for labels x-, y-axes and colorbar
ticksize = 12.5 # for label ticks
titlesize = 20 # for title
zPlotRange = [0,0.07] # Which z values do you want to plot?  
yCoordinates = [0,3]  # Which range of y coordinates should the DEM contain? # Default [0,3]
xCoordinates = [0,20] # Which range of x coordinates should the DEM contain? # Default [0,20]
# Plot boolean dry/wet map from weirDEMdepthMap
ax = plt.figure(figsize=figsize)
plt.pcolormesh(xx, yy, mask062_4636,
               rasterized=True)
plt.xlim(xCoordinates[0],xCoordinates[1])
plt.ylim(yCoordinates[0],yCoordinates[1])
plt.tight_layout()
plt.tick_params(labelsize=ticksize)
plt.xlabel('x (m)', size=labelsize)
plt.ylabel('y (m)', size=labelsize)
# Add colorbar
clb = plt.colorbar(location='right', pad = 0.01)
clb.set_label('Dry == 0    Wet == 1', size=labelsize)
clb.ax.tick_params(labelsize=ticksize)
# Store image

#%% Load Dems and cunstruct weirDEMDepthMaps 
# Exp062
weirDems062 = []
weirDems062_1 = []
weirDems062_2 = [] # Stacked
dryWetMaps062 = []

for i in range(len(demFiles062)):
    # Current DEM
    demFile = demFiles062[i]
    # Load DEM
    netCDFFile = nc.Dataset(demFolder062 + '\\' + demFile)
    # Extract the relevant information
    zData = np.array(netCDFFile.get_variables_by_attributes(name='Z-axis')[0])
    # We can close the file again
    netCDFFile.close()
    # Apply mask to DEM
    # Make zData two-dimensional
    zData = zData[:,:,0].reshape((zData.shape[0],zData.shape[1]))
    # Apply a vertical flip due to inversed x-axes between mask and zData
    # Mask is different for '4636'
    if Exp062_weirs[i] == '4636':
        zMask062 = zData*np.flip(mask062_4636,axis=0)
    else:    
        zMask062 = zData*np.flip(mask062,axis=0)
    
    # Construct weirDEMDepthMap
    # Make depth maps from DEM and weir levels
    waterLevel = float(Exp062_weirs[i])/100000 + weirOffset #- 0.001
    weirDEMdepthMap062 = waterLevel - zMask062
    # Set negative values to zero
    weirDEMdepthMap062[weirDEMdepthMap062 < 0] = 0

    # Apply mask
    weirDEMdepthMap0621 = weirDEMdepthMap062.copy()
    if Exp062_weirs[i] == '4636':
        weirDEMdepthMap0621[np.flip(mask062_4636,axis=0) == 0] = 0
        weirDEMdepthMap062[np.flip(mask062_4636,axis=0) == 0] = np.nan
    else:
        weirDEMdepthMap0621[np.flip(mask062,axis=0) == 0] = 0
        weirDEMdepthMap062[np.flip(mask062,axis=0) == 0] = np.nan

    # Construct boolean dry/wet map from weirDEMdepthMap
    weirDEMDryWetMap062 = weirDEMdepthMap062.copy()
    weirDEMDryWetMap0621 = weirDEMdepthMap0621.copy()
    # Set positive values to one
    weirDEMDryWetMap062[weirDEMDryWetMap062 > 0] = 1
    weirDEMDryWetMap0621[weirDEMDryWetMap0621 > 0] = 1

    # Flip again for relating to ortho depth map
    weirDEMDryWetMap0623 = weirDEMDryWetMap062.copy()
    weirDEMDryWetMap0623 = np.flip(weirDEMDryWetMap0623,axis=0)
    weirDEMdepthMap0623 = weirDEMdepthMap062.copy()
    weirDEMdepthMap0623 = np.flip(weirDEMdepthMap0623,axis=0)

    # Stack information
    weirDEMdepthMapStacked062 = weirDEMdepthMap0623.reshape(-1)
    weirDEMDryWetMapStacked062 = weirDEMDryWetMap0623.reshape(-1)
    # Remove nans
    weirDEMdepthMapStacked062 = weirDEMdepthMapStacked062[~np.isnan(weirDEMdepthMapStacked062)]
    weirDEMDryWetMapStacked062 = weirDEMDryWetMapStacked062[~np.isnan(weirDEMDryWetMapStacked062)]
    # Flip and set to int
    weirDEMDryWetMapStacked062 = weirDEMDryWetMapStacked062.astype(int)
    
    # Append all relevant data
    weirDems062.append(weirDEMdepthMap062)
    weirDems062_1.append(weirDEMdepthMap0621)
    weirDems062_2.append(weirDEMdepthMapStacked062)
    dryWetMaps062.append(weirDEMDryWetMapStacked062)
    
    del demFile, netCDFFile, zData, zMask062, weirDEMdepthMap062, weirDEMdepthMap0621, weirDEMDryWetMap0623, weirDEMDryWetMapStacked062, weirDEMdepthMapStacked062
    del weirDEMDryWetMap0621, weirDEMDryWetMap062, weirDEMdepthMap0623
    
# Exp077
# Current DEM
demFile = demFile077

# Load DEM
netCDFFile = nc.Dataset(demFolder077 + '\\' + demFile)
# Extract the relevant information
zData = np.array(netCDFFile.get_variables_by_attributes(name='Z-axis')[0])
# We can close the file again
netCDFFile.close()
# Apply mask to DEM
# Make zData two-dimensional
zData = zData[:,:,0].reshape((zData.shape[0],zData.shape[1]))
# Apply a vertical flip due to inversed x-axes between mask and zData
zMask077 = zData*np.flip(mask077,axis=0)

# Construct weirDEMDepthMap
# Make depth maps from DEM and weir levels
waterLevel = float(Exp077_weir)/100000 + weirOffset #- 0.001
weirDEMdepthMap077 = waterLevel - zMask077
# Set negative values to zero
weirDEMdepthMap077[weirDEMdepthMap077 < 0] = 0

# Apply mask
weirDEMdepthMap0771 = weirDEMdepthMap077.copy()
weirDEMdepthMap0771[np.flip(mask077,axis=0) == 0] = 0
weirDEMdepthMap077[np.flip(mask077,axis=0) == 0] = np.nan

# Construct boolean dry/wet map from weirDEMdepthMap
weirDEMDryWetMap077 = weirDEMdepthMap077.copy()
weirDEMDryWetMap0771 = weirDEMdepthMap0771.copy()
# Set positive values to one
weirDEMDryWetMap077[weirDEMDryWetMap077 > 0] = 1
weirDEMDryWetMap0771[weirDEMDryWetMap0771 > 0] = 1

# Flip again for relating to ortho depth map
weirDEMDryWetMap0773 = weirDEMDryWetMap077.copy()
weirDEMDryWetMap0773 = np.flip(weirDEMDryWetMap0773,axis=0)
weirDEMdepthMap0773 = weirDEMdepthMap077.copy()
weirDEMdepthMap0773 = np.flip(weirDEMdepthMap0773,axis=0)

# Stack information
weirDEMdepthMapStacked077 = weirDEMdepthMap0773.reshape(-1)
weirDEMDryWetMapStacked077 = weirDEMDryWetMap0773.reshape(-1)
# Remove nans
weirDEMdepthMapStacked077 = weirDEMdepthMapStacked077[~np.isnan(weirDEMdepthMapStacked077)]
weirDEMDryWetMapStacked077 = weirDEMDryWetMapStacked077[~np.isnan(weirDEMDryWetMapStacked077)]
# Set to int
weirDEMDryWetMapStacked077 = weirDEMDryWetMapStacked077.astype(int)

weirDems077 = weirDEMdepthMap077
weirDems077_1 = weirDEMdepthMap0771
weirDems077_2 = weirDEMdepthMapStacked077
dryWetMaps077 = weirDEMDryWetMapStacked077

del demFile, netCDFFile, zData, zMask077, weirDEMdepthMap077, weirDEMdepthMap0771, weirDEMDryWetMap0773, weirDEMDryWetMapStacked077, weirDEMdepthMapStacked077
del weirDEMDryWetMap0771, weirDEMDryWetMap077, weirDEMdepthMap0773

#%% Prepare orthos 
# Exp062
orthos062 = []
weirDems062_stacked = []
dryWet062_stacked = []

# Loop through the orthos for Exp062
for i in range(len(Exp062_orthos)):
    # What is the current ortho folder
    currentOrthoFolder = prepFolder + '\\Orthos\\' + Exp062_orthos[i]
    # Make list
    orthoList = [file for file in os.listdir(currentOrthoFolder) if not file.endswith('Thumbs.db')]
    orthos = []
    
    for ortho in orthoList:
        # Load ortho
        currentOrtho = cv2.imread(currentOrthoFolder + '\\' + ortho)
        # Apply mask
        if Exp062_weirs[i] == '4636':
            maskedOrtho = np.array(currentOrtho*orthoMask3062_4636,dtype='uint8')
        else:
            maskedOrtho = np.array(currentOrtho*orthoMask3062,dtype='uint8')
        # Copy
        maskedOrtho0 = maskedOrtho.copy()
        maskedOrtho1 = maskedOrtho.copy()
        maskedOrtho1[maskedOrtho1>0] = 1
        # Set to floats
        maskedOrtho0 = maskedOrtho0.astype(float)
        maskedOrtho1 = maskedOrtho1.astype(float)
        # Apply Gaussion blur
        dst0 = cv2.GaussianBlur(maskedOrtho0 ,(smGsettings[0],smGsettings[1]),smGsettings[2],smGsettings[3],borderType = borderTypeO)
        dst1 = cv2.GaussianBlur(maskedOrtho1 ,(smGsettings[0],smGsettings[1]),smGsettings[2],smGsettings[3],borderType = borderTypeO)
        # Divide
        dst = dst0/dst1 
        # Apply mask
        dst[maskedOrtho1==0]=np.nan
        # Round values to nearest integer
        dst = np.rint(dst)
        # Set back to uint8
        gausOrtho = dst.astype(np.uint8)
        # Stack information
        orthoStacked = gausOrtho.reshape(-1,1,np.shape(gausOrtho)[-1])
        # Remove masked area (all zeros)
        orthoStacked = orthoStacked[orthoStacked > 0]
        orthoStacked = orthoStacked.reshape(int(len(orthoStacked)/3),1,3)
        # Append the bingo2 chart of colorspaces
        # LAB
        orthoStacked = np.append(orthoStacked,rgb_to_LAB(orthoStacked),axis=2)
        # Luv
        orthoStacked = np.append(orthoStacked,rgb_to_Luv(orthoStacked[:,:,:3]),axis=2)
        # YCrCb
        orthoStacked = np.append(orthoStacked,rgb_to_YCrCb(orthoStacked[:,:,:3]),axis=2)
        # HSV
        orthoStacked = np.append(orthoStacked,rgb_to_HSV(orthoStacked[:,:,:3]),axis=2)
        # Now Restructure to 2D
        orthoStacked = orthoStacked.reshape(-1,np.shape(orthoStacked)[-1])
        # Append to orthos
        orthos.append(orthoStacked)
        
        # Append stack dem and boolMap
        weirDems062_stacked.append(weirDems062_2[i])
        dryWet062_stacked.append(dryWetMaps062[i])
        
    orthos062.append(orthos)      
    
    del ortho, orthos, currentOrtho, maskedOrtho, maskedOrtho0, maskedOrtho1, dst0, dst1, dst, gausOrtho, orthoStacked, orthoList

# Exp077
currentOrthoFolder = prepFolder + '\\Orthos\\' + Exp077_12675_6700
# Make list
orthoList = [file for file in os.listdir(currentOrthoFolder) if not file.endswith('Thumbs.db')]
orthos = []
weirDems077_stacked = []
dryWet077_stacked = []

for ortho in orthoList:
    # Load ortho
    currentOrtho = cv2.imread(currentOrthoFolder + '\\' + ortho)
    # Apply mask
    maskedOrtho = np.array(currentOrtho*orthoMask3077,dtype='uint8')
    # Copy
    maskedOrtho0 = maskedOrtho.copy()
    maskedOrtho1 = maskedOrtho.copy()
    maskedOrtho1[maskedOrtho1>0] = 1
    # Set to floats
    maskedOrtho0 = maskedOrtho0.astype(float)
    maskedOrtho1 = maskedOrtho1.astype(float)
    # Apply Gaussion blur
    dst0 = cv2.GaussianBlur(maskedOrtho0 ,(smGsettings[0],smGsettings[1]),smGsettings[2],smGsettings[3],borderType = borderTypeO)
    dst1 = cv2.GaussianBlur(maskedOrtho1 ,(smGsettings[0],smGsettings[1]),smGsettings[2],smGsettings[3],borderType = borderTypeO)
    # Divide
    dst = dst0/dst1
    # Apply mask
    dst[maskedOrtho1==0]=np.nan
    # Round values to nearest integer
    dst = np.rint(dst)
    # Set back to uint8
    gausOrtho = dst.astype(np.uint8)
    # Stack information
    orthoStacked = gausOrtho.reshape(-1,1,np.shape(gausOrtho)[-1])
    # Remove masked area (all zeros)
    orthoStacked = orthoStacked[orthoStacked > 0]
    orthoStacked = orthoStacked.reshape(int(len(orthoStacked)/3),1,3)
    # Append the bingo2 chart of colorspaces
    # LAB
    orthoStacked = np.append(orthoStacked,rgb_to_LAB(orthoStacked),axis=2)
    # Luv
    orthoStacked = np.append(orthoStacked,rgb_to_Luv(orthoStacked[:,:,:3]),axis=2)
    # YCrCb
    orthoStacked = np.append(orthoStacked,rgb_to_YCrCb(orthoStacked[:,:,:3]),axis=2)
    # HSV
    orthoStacked = np.append(orthoStacked,rgb_to_HSV(orthoStacked[:,:,:3]),axis=2)
    # Now Restructure to 2D
    orthoStacked = orthoStacked.reshape(-1,np.shape(orthoStacked)[-1])
    # Append to orthos
    orthos.append(orthoStacked)
    
    # Append stack dem
    weirDems077_stacked.append(weirDems077_2)
    dryWet077_stacked.append(dryWetMaps077)
    
orthos077 = orthos  
    
del ortho, orthos, currentOrtho, maskedOrtho, maskedOrtho0, maskedOrtho1, dst0, dst1, dst, gausOrtho, orthoStacked, orthoList

# Stack all data
stack062 = []
for orthoList in orthos062:
    stack062.append(np.concatenate(orthoList,axis=0))
stacks062 = np.concatenate(stack062,axis=0)    
stacks077 = np.concatenate(orthos077,axis=0)
stackDems062 = np.concatenate(weirDems062_stacked,axis=0)    
stackDems077 = np.concatenate(weirDems077_stacked,axis=0)  
stackBool062 = np.concatenate(dryWet062_stacked,axis=0)  
stackBool077 = np.concatenate(dryWet077_stacked,axis=0)  

del  weirDems077_2, weirDems062_2, stack062, orthos077, weirDems062_stacked, weirDems077_stacked, dryWet062_stacked, dryWet077_stacked, dryWetMaps062, dryWetMaps077

print('All data is prepared for ML application and validation')

#%% Where to store the relevant information
recalls062 = []; recalls077 = []
precisions062 = []; precisions077 = []
F1scores062 = []; F1scores077 = []
peirceSkillScores062 = []; peirceSkillScores077 = []
rmse062 = []; rmse077 = []
bias062 = []; bias077 = []
bias2prmse062 = []; bias2prmse077 = []
falseWetPredDepths062 = []; falseWetPredDepths077 = []
falseDryActualDepths062 = []; falseDryActualDepths077 = []
trueWetPredDepths062 = []; trueWetPredDepths077 = []
trueWetActualDepths062 = []; trueWetActualDepths077 = []

#%% Loop through the concentrations
for concentration in concentrations:
    print('Computing validations for concentration ' + concentration)
    
    # Append concentration
    recalls062.append('Conc' + concentration); recalls077.append('Conc' + concentration)
    precisions062.append('Conc' + concentration); precisions077.append('Conc' + concentration)
    F1scores062.append('Conc' + concentration); F1scores077.append('Conc' + concentration)
    peirceSkillScores062.append('Conc' + concentration); peirceSkillScores077.append('Conc' + concentration)
    rmse062.append('Conc' + concentration); rmse077.append('Conc' + concentration)
    bias062.append('Conc' + concentration); bias077.append('Conc' + concentration)
    bias2prmse062.append('Conc' + concentration); bias2prmse077.append('Conc' + concentration)
    falseWetPredDepths062.append('Conc' + concentration); falseWetPredDepths077.append('Conc' + concentration)
    falseDryActualDepths062.append('Conc' + concentration); falseDryActualDepths077.append('Conc' + concentration)
    trueWetPredDepths062.append('Conc' + concentration); trueWetPredDepths077.append('Conc' + concentration)
    trueWetActualDepths062.append('Conc' + concentration); trueWetActualDepths077.append('Conc' + concentration)

    # Load the current models
    clfFile = modelFolder + '\\Conc' + concentration + 'Clf.joblib'
    regrFile = modelFolder + '\\Conc' + concentration + 'Regr.joblib'
    clf = joblib.load(clfFile)
    print("Classifier model loaded for Conc" + concentration)
    regr = joblib.load(regrFile)
    print("Regression model loaded for Conc" + concentration)
    
    #%% Apply to Exp062
    # Dry/wet classifier
    clfResult062 = clf.predict(stacks062)    
    # Regressor
    regrResult062 = regr.predict(stacks062)
    # Combine Results
    combResult062 = clfResult062*regrResult062
    combResultWet062 = combResult062[clfResult062==1]
    waterDepthsWet062 = stackDems062[clfResult062==1]
    
    # We're also interested in the indices of false positives and false negatives
    falseWetClf062 = np.where(np.logical_and(clfResult062 == 1,stackBool062 == 0))[0]
    falseDryClf062 = np.where(np.logical_and(clfResult062 == 0,stackBool062 == 1))[0]
    # And for the comparison between predicted true depths at actual depths
    trueWetClf062 = np.where(np.logical_and(clfResult062 == 1,stackBool062 == 1))[0]
    
    # Calculate relevant scores of the clf model
    recalls062.append(metrics.recall_score(stackBool062,clfResult062,average=None))
    precisions062.append(metrics.precision_score(stackBool062,clfResult062,average=None))
    F1scores062.append(metrics.f1_score(stackBool062,clfResult062,average=None))

    # We're also interested in Peirce skill score, for which we use a different package
    xrclfResultWet062 = xr.DataArray(clfResult062)
    xrbinaryMapWet062 = xr.DataArray(stackBool062)
    contingency_manager_wet062 = scores.categorical.BinaryContingencyManager(xrclfResultWet062, xrbinaryMapWet062)
    peirceSkillScores062.append(contingency_manager_wet062.peirce_skill_score().values) # Peirce skill score is equal for dry and wet
        
    # RMSE
    rmse062.append(np.sqrt(metrics.mean_squared_error(waterDepthsWet062, combResultWet062)))
    
    # Bias # == mean error
    bias062.append(-1*np.mean(waterDepthsWet062 - combResultWet062))
    
    # Coefficient bias2 as fraction of rmse
    bias2prmse062.append((bias062[-1]**2)/(rmse062[-1]**2))
    
    # Now store the predicted depths of false wet cells and true depths of false dry cells
    falseWetPredDepths062.append(combResult062[falseWetClf062])
    falseDryActualDepths062.append(stackDems062[falseDryClf062])
    
    # As well as the true depths (for heatmaps)
    trueWetPredDepths062.append(combResult062[trueWetClf062])
    trueWetActualDepths062.append(stackDems062[trueWetClf062])

    print("All data has been validated for Exp062 and Conc" + concentration)
    
    #%% Repeat for Exp077
    # Dry/wet classifier
    clfResult077 = clf.predict(stacks077)    
    # Regressor
    regrResult077 = regr.predict(stacks077)
    # Combine Results
    combResult077 = clfResult077*regrResult077
    combResultWet077 = combResult077[clfResult077==1]
    waterDepthsWet077 = stackDems077[clfResult077==1]
    
    # We're also interested in the indices of false positives and false negatives
    falseWetClf077 = np.where(np.logical_and(clfResult077 == 1,stackBool077 == 0))[0]
    falseDryClf077 = np.where(np.logical_and(clfResult077 == 0,stackBool077 == 1))[0]
    # And for the comparison between predicted true depths at actual depths
    trueWetClf077 = np.where(np.logical_and(clfResult077 == 1,stackBool077 == 1))[0]
    
    # Calculate relevant scores of the clf model
    recalls077.append(metrics.recall_score(stackBool077,clfResult077,average=None))
    precisions077.append(metrics.precision_score(stackBool077,clfResult077,average=None))
    F1scores077.append(metrics.f1_score(stackBool077,clfResult077,average=None))

    # We're also interested in Peirce skill score, for which we use a different package
    xrclfResultWet077 = xr.DataArray(clfResult077)
    xrbinaryMapWet077 = xr.DataArray(stackBool077)
    contingency_manager_wet077 = scores.categorical.BinaryContingencyManager(xrclfResultWet077, xrbinaryMapWet077)
    peirceSkillScores077.append(contingency_manager_wet077.peirce_skill_score().values) # Peirce skill score is equal for dry and wet
        
    # RMSE
    rmse077.append(np.sqrt(metrics.mean_squared_error(waterDepthsWet077, combResultWet077)))
    
    # Bias # == mean error
    bias077.append(-1*np.mean(waterDepthsWet077 - combResultWet077))
    
    # Coefficient bias2 as fraction of rmse
    bias2prmse077.append((bias077[-1]**2)/(rmse077[-1]**2))
    
    # Now store the predicted depths of false wet cells and true depths of false dry cells
    falseWetPredDepths077.append(combResult077[falseWetClf077])
    falseDryActualDepths077.append(stackDems077[falseDryClf077])
    
    # As well as the true depths (for heatmaps)
    trueWetPredDepths077.append(combResult077[trueWetClf077])
    trueWetActualDepths077.append(stackDems077[trueWetClf077])

    print("All data has been validated for Exp077 and Conc" + concentration)
    
#%% Write all information to CSV files
# Exp062
# Single variable data of training datasets
with open(prepFolder + '\\Fig6_validationData_Exp062.csv', 'w', newline='') as csvFile0620:
    csvFile0620 = csv.writer(csvFile0620,delimiter=",")
    csvFile0620.writerow([['Model_Conc'],['F1Wet'],['F1Dry'],['Peirce'],['RecallWet'],['RecallDry'],
                          ['PrecisionWet'],['PrecisionDry'],['RMSE'],['Bias'],['Bias2FracMSE']])
    for i in range(0,len(F1scores062),2):
        csvFile0620.writerow([[F1scores062[i]],[F1scores062[i+1][0]],[F1scores062[i+1][1]],[np.atleast_1d(peirceSkillScores062[i+1])[0]],
                              [recalls062[i+1][0]],[recalls062[i+1][1]],[precisions062[i+1][0]],[precisions062[i+1][1]],
                              [rmse062[i+1]],[bias062[i+1]],[bias2prmse062[i+1]]])
    
# Predicted water depths from falsely predicted wet cells
with open(prepFolder + '\\Fig6_falseWetPredDepth_Exp062.csv', 'w', newline='') as csvFile0621:
    csvFile0621 = csv.writer(csvFile0621,delimiter=",")
    csvFile0621.writerow([['Model_Conc'],['values']])
    for i in range(0,len(falseWetPredDepths062),2):
        # Trainig datasets
        csvFile0621.writerow([str(falseWetPredDepths062[i])])
        csvFile0621.writerow(falseWetPredDepths062[i+1])
        
# True water depths from falsely predicted dry cells
with open(prepFolder + '\\Fig6_falseDryTrueDepths_Exp062.csv', 'w', newline='') as csvFile0622:
    csvFile0622 = csv.writer(csvFile0622,delimiter=",")
    csvFile0622.writerow([['Model_Conc'],['values']])
    for i in range(0,len(falseDryActualDepths062),2):
        # Trainig datasets
        csvFile0622.writerow([str(falseDryActualDepths062[i])])
        csvFile0622.writerow(falseDryActualDepths062[i+1])

# Predicted water depths from correctly predicted wet cells
with open(prepFolder + '\\Fig6_TrueWetPredDepths_Exp062.csv', 'w', newline='') as csvFile0623:
    csvFile0623 = csv.writer(csvFile0623,delimiter=",")
    csvFile0623.writerow([['Model_Conc'],['values']])
    for i in range(0,len(trueWetPredDepths062),2):
        # Trainig datasets
        csvFile0623.writerow([str(trueWetPredDepths062[i])])
        csvFile0623.writerow(trueWetPredDepths062[i+1])
        
# True water depths from correctly predicted dry cells
with open(prepFolder + '\\Fig6_TrueWetTrueDepths_Exp062.csv', 'w', newline='') as csvFile0624:
    csvFile0624 = csv.writer(csvFile0624,delimiter=",")
    csvFile0624.writerow([['Model_Conc'],['values']])
    for i in range(0,len(trueWetActualDepths062),2):
        # Trainig datasets
        csvFile0624.writerow([str(trueWetActualDepths062[i])])
        csvFile0624.writerow(trueWetActualDepths062[i+1])
        
#%% Exp077
# Single variable data of training datasets
with open(prepFolder + '\\Fig6_validationData_Exp077.csv', 'w', newline='') as csvFile0770:
    csvFile0770 = csv.writer(csvFile0770,delimiter=",")
    csvFile0770.writerow([['Model_Conc'],['F1Wet'],['F1Dry'],['Peirce'],['RecallWet'],['RecallDry'],
                          ['PrecisionWet'],['PrecisionDry'],['RMSE'],['Bias'],['Bias2FracMSE']])
    for i in range(0,len(F1scores077),2):
        csvFile0770.writerow([[F1scores077[i]],[F1scores077[i+1][0]],[F1scores077[i+1][1]],[np.atleast_1d(peirceSkillScores077[i+1])[0]],
                              [recalls077[i+1][0]],[recalls077[i+1][1]],[precisions077[i+1][0]],[precisions077[i+1][1]],
                              [rmse077[i+1]],[bias077[i+1]],[bias2prmse077[i+1]]])
    
# Predicted water depths from falsely predicted wet cells
with open(prepFolder + '\\Fig6_falseWetPredDepth_Exp077.csv', 'w', newline='') as csvFile0771:
    csvFile0771 = csv.writer(csvFile0771,delimiter=",")
    csvFile0771.writerow([['Model_Conc'],['values']])
    for i in range(0,len(falseWetPredDepths077),2):
        # Trainig datasets
        csvFile0771.writerow([str(falseWetPredDepths077[i])])
        csvFile0771.writerow(falseWetPredDepths077[i+1])
        
# True water depths from falsely predicted dry cells
with open(prepFolder + '\\Fig6_falseDryTrueDepths_Exp077.csv', 'w', newline='') as csvFile0772:
    csvFile0772 = csv.writer(csvFile0772,delimiter=",")
    csvFile0772.writerow([['Model_Conc'],['values']])
    for i in range(0,len(falseDryActualDepths077),2):
        # Trainig datasets
        csvFile0772.writerow([str(falseDryActualDepths077[i])])
        csvFile0772.writerow(falseDryActualDepths077[i+1])

# Predicted water depths from correctly predicted wet cells
with open(prepFolder + '\\Fig6_TrueWetPredDepths_Exp077.csv', 'w', newline='') as csvFile0773:
    csvFile0773 = csv.writer(csvFile0773,delimiter=",")
    csvFile0773.writerow([['Model_Conc'],['values']])
    for i in range(0,len(trueWetPredDepths077),2):
        # Trainig datasets
        csvFile0773.writerow([[str(trueWetPredDepths077[i])],['Training']])
        csvFile0773.writerow(trueWetPredDepths077[i+1])
        
# True water depths from correctly predicted dry cells
with open(prepFolder + '\\Fig6_TrueWetTrueDepths_Exp077.csv', 'w', newline='') as csvFile0774:
    csvFile0774 = csv.writer(csvFile0774,delimiter=",")
    csvFile0774.writerow([['Model_Conc'],['values']])
    for i in range(0,len(trueWetActualDepths077),2):
        # Trainig datasets
        csvFile0774.writerow([str(trueWetActualDepths077[i])])
        csvFile0774.writerow(trueWetActualDepths077[i+1])
        

#%% How long did the run take?
end = datetime.datetime.now()
td = round((end - start).total_seconds())
print("Script running time = " + str(datetime.timedelta(seconds=td)))
