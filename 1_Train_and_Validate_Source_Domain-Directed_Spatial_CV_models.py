'''
Code to construct and validate Classifier and Regression Random Forest Models from the overhead orthomaics
following the Source Domain-Directed Cross-Validation approach
The input is the same Training imagery as for the original Random Forest models
Relevant input includes mask (.csv) files describing the exclusive camera zones and buffers

Python script written by Eise W. Nota (finalized JUNE 2026)

'''

#%% Clearing all variables
from IPython import get_ipython
get_ipython().magic('reset -sf') 

#%% Import packages
import cv2
import os
import numpy as np
import pandas as pd
import netCDF4 as nc
import datetime
import csv
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import joblib
import scores.categorical
import xarray as xr

# Set base directory of the network drive
pwd =  r"..."

start = datetime.datetime.now()

#%% Defining the variables
# General
Expnr = '058'                                   # Experiment number
baseModel = '11'
orthoRes = 25 # mm
DEMsurvey = 'dry'
concentrations = ['0','1','2','3','4','5']
whichMask = '2' # either '1' (excluding  actuators) or '2' (including actuators)
# What are the max features for each concentration?
max_features_Clf = [None,None,None,None,None,None]
max_features_Regr = [4,3,4,5,4,4]
# Gaussian smoothener settings
smGsettings = [5,5,0,0]

# What are the relevant bands
input_data_sets = ['RGB', 'LAB', 'Luv', 'YCrCb', 'HSV']
input_data_names = ['Red','Green','Blue']
for i in range(1,len(input_data_sets)):
    currentSet = input_data_sets[i]
    if currentSet == 'YCrCb':
        input_data_names.append('YCrCb-Y')
        input_data_names.append('YCrCb-Cr')
        input_data_names.append('YCrCb-Cb')
    else:
        input_data_names.append(currentSet + '-' + currentSet[0])
        input_data_names.append(currentSet + '-' + currentSet[1])
        input_data_names.append(currentSet + '-' + currentSet[2])
        
# Define weir offset
finalWeirLevel = 0.08613
differenceWaterLevel = 0.00
demfloodLevel = 0.102
weirOffset = round((demfloodLevel - differenceWaterLevel) - finalWeirLevel,5)

#%% Determine mask and DEM directories
demFolder = pwd + r'\\dem\\'
maskFolder = pwd + r'\\mask\\'
modelCVFolder = pwd + r'\\modelsCV\\'
camMaskFolder = pwd + r'\\camMasks\\'
validationFolder = pwd + r'\\validation\\Source_Domain-Directed_CV_on_Training\\'
expCalibFolder = pwd + r'\\smG\\'

#%% Load DEM
DEM_name = 'Exp' + Expnr + '_' + DEMsurvey  + '_LaserScan_DEM.nc'
    
netCDFFile = nc.Dataset(demFolder + '\\' + DEM_name)
# Extract the relevant information
zData = np.array(netCDFFile.get_variables_by_attributes(name='Z-axis')[0])
# We can close the file again
netCDFFile.close()

# Make zData two-dimensional
zData = zData[:,:,0].reshape((zData.shape[0],zData.shape[1]))

#%% Let's apply mask
# Load mask
pdMask = pd.read_csv(maskFolder + '\\mask' + whichMask + '.csv', delimiter=',', header=None)     
# Make array
mask = np.array(pdMask)
# Set zeros to NaN
mask = mask.astype('float')
mask[mask == 0] = np.nan

# Apply mask to DEM
# Apply a vertical flip due to inversed x-axes between mask and zData
zMask = zData*np.flip(mask,axis=0)
zMask = np.flip(zMask,axis=0)

del(netCDFFile, zData, pdMask, DEM_name)

#%% Load cam and buffer masks
maskB1_2 = np.array(pd.read_csv(camMaskFolder + '\\maskBuffer1_2.csv', delimiter=',', header=None))
maskB2_3 = np.array(pd.read_csv(camMaskFolder + '\\maskBuffer2_3.csv', delimiter=',', header=None))
maskB3_4 = np.array(pd.read_csv(camMaskFolder + '\\maskBuffer3_4.csv', delimiter=',', header=None))
maskB4_5 = np.array(pd.read_csv(camMaskFolder + '\\maskBuffer4_5.csv', delimiter=',', header=None))
maskB5_6 = np.array(pd.read_csv(camMaskFolder + '\\maskBuffer5_6.csv', delimiter=',', header=None)) 
maskB6_7 = np.array(pd.read_csv(camMaskFolder + '\\maskBuffer6_7.csv', delimiter=',', header=None)) 
maskC1 = np.array(pd.read_csv(camMaskFolder + '\\maskCam1.csv', delimiter=',', header=None))
maskC2 = np.array(pd.read_csv(camMaskFolder + '\\maskCam2.csv', delimiter=',', header=None))
maskC3 = np.array(pd.read_csv(camMaskFolder + '\\maskCam3.csv', delimiter=',', header=None))
maskC4 = np.array(pd.read_csv(camMaskFolder + '\\maskCam4.csv', delimiter=',', header=None)) 
maskC5 = np.array(pd.read_csv(camMaskFolder + '\\maskCam5.csv', delimiter=',', header=None)) 
maskC6 = np.array(pd.read_csv(camMaskFolder + '\\maskCam6.csv', delimiter=',', header=None)) 
maskC7 = np.array(pd.read_csv(camMaskFolder + '\\maskCam7.csv', delimiter=',', header=None)) 

# Stack
masksB = [maskB1_2,maskB1_2+maskB2_3,maskB2_3+maskB3_4,maskB4_5,maskB4_5+maskB5_6,maskB5_6+maskB6_7,maskB6_7]
masksC = [maskC1,maskC2,maskC3,maskC4,maskC5,maskC6,maskC7]
del(maskB1_2,maskB2_3,maskB3_4,maskB4_5,maskB5_6,maskB6_7)
del(maskC1,maskC2,maskC3,maskC4,maskC5,maskC6,maskC7)
# Also combine # only for validation
masksBC = []
for i in range(len(masksB)):
    masksBC.append(masksB[i]+masksC[i])

# Inverted masks are important for validation
masksBi = masksB.copy()
masksCi = masksC.copy()

# Loop through masksB
for i in range(len(masksB)):
    # Set ones to NaN and zeros to ones
    masksB[i] = masksB[i].astype('float')
    masksB[i][masksB[i] == 1] = np.nan
    masksB[i][masksB[i] == 0] = 1
    # And other away around for the inverted
    masksBi[i] = masksBi[i].astype('float')
    masksBi[i][masksBi[i] == 0] = np.nan
    # Remove rows and columns that have all zeroes in zMask
    masksB[i] = masksB[i][~np.isnan(zMask).all(axis=1),:] # Rows
    masksB[i] = masksB[i][:,~np.isnan(zMask).all(axis=0)] # Columns
    masksBi[i] = masksBi[i][~np.isnan(zMask).all(axis=1),:] # Rows
    masksBi[i] = masksBi[i][:,~np.isnan(zMask).all(axis=0)] # Columns
    # Convert to 3D
    masksB[i] = masksB[i][:, :, np.newaxis]
    masksBi[i] = masksBi[i][:, :, np.newaxis]
    # Continue with masksC
    # Set ones to NaN and zeros to ones
    masksC[i] = masksC[i].astype('float')
    masksC[i][masksC[i] == 1] = np.nan
    masksC[i][masksC[i] == 0] = 1
    # And other away around for the inverted
    masksCi[i] = masksCi[i].astype('float')
    masksCi[i][masksCi[i] == 0] = np.nan
    # Remove rows and columns that have all zeroes in zMask
    masksC[i] = masksC[i][~np.isnan(zMask).all(axis=1),:] # Rows
    masksC[i] = masksC[i][:,~np.isnan(zMask).all(axis=0)] # Columns
    masksCi[i] = masksCi[i][~np.isnan(zMask).all(axis=1),:] # Rows
    masksCi[i] = masksCi[i][:,~np.isnan(zMask).all(axis=0)] # Columns
    # Convert to 3D
    masksC[i] = masksC[i][:, :, np.newaxis]
    masksCi[i] = masksCi[i][:, :, np.newaxis]
    # Continue with masksBC (only validation)
    # Set zeros to NaN
    masksBC[i] = masksBC[i].astype('float')
    masksBC[i][masksBC[i] == 0] = np.nan
    # Remove rows and columns that have all zeroes in zMask
    masksBC[i] = masksBC[i][~np.isnan(zMask).all(axis=1),:] # Rows
    masksBC[i] = masksBC[i][:,~np.isnan(zMask).all(axis=0)] # Columns
    # Convert to 3D
    masksBC[i] = masksBC[i][:, :, np.newaxis]
    
# Remove rows and columns that have all zeroes
zMask = zMask[~np.isnan(zMask).all(axis=1),:] # Rows
zMask = zMask[:,~np.isnan(zMask).all(axis=0)] # Columns

#%% Determine root directories from the dry orthos
dryOrthoFolderO0 = expCalibFolder + r'\\RGB\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\Dry'
dryOrthoFolderO1 = expCalibFolder + r'\\LAB\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\Dry'
dryOrthoFolderO2 = expCalibFolder + r'\\Luv\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\Dry'
dryOrthoFolderO4 = expCalibFolder + r'\\YCrCb\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\Dry'
dryOrthoFolderO5 = expCalibFolder + r'\\HSV\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\Dry'

#%% Loop over concentrations
for c in range(len(concentrations)):
    concentration = concentrations[c]
    max_feature_Clf = max_features_Clf[c]
    max_feature_Regr = max_features_Regr[c]
    print('Computing validations for concentration ' + concentration)
    
    # Determine model directories
    clfFolder = modelCVFolder + r"\\models\\Conc" + concentration + "\\clf\\"
    regrFolder = modelCVFolder + r"\\models\\Conc" + concentration + "\\regr\\"
    valFolder = validationFolder + r"\\Conc" + concentration + "\\"
   
    # Create folder if it doesn't exist
    if not os.path.exists(clfFolder):
        os.makedirs(clfFolder)
    if not os.path.exists(regrFolder):
        os.makedirs(regrFolder)
    if not os.path.exists(valFolder):
        os.makedirs(valFolder)
   
    # As well as from the training dataset
    orthoFolderO0 = expCalibFolder + r'\\RGB\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\Conc' + concentration
    orthoFolderO1 = expCalibFolder + r'\\LAB\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\Conc' + concentration
    orthoFolderO2 = expCalibFolder + r'\\Luv\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\Conc' + concentration
    orthoFolderO4 = expCalibFolder + r'\\YCrCb\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\Conc' + concentration
    orthoFolderO5 = expCalibFolder + r'\\HSV\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\Conc' + concentration
    
    #%% We're going to make the models from the training datasets and validate them on both training and validation datasets
    # Where to store information
    ortho0ints = []
    orthosO = []
    
    # Load the Original information as the training datasets for the Cross-validation
    # Load dry orthos
    dryOrtho0 = [file for file in os.listdir(dryOrthoFolderO0) if not file.endswith('Thumbs.db')][0]
    dryOrtho0 = cv2.imread(dryOrthoFolderO0 + '\\' + dryOrtho0)
    dryOrtho1 = [file for file in os.listdir(dryOrthoFolderO1) if not file.endswith('Thumbs.db')][0]
    dryOrtho1 = cv2.imread(dryOrthoFolderO1 + '\\' + dryOrtho1)
    dryOrtho2 = [file for file in os.listdir(dryOrthoFolderO2) if not file.endswith('Thumbs.db')][0]
    dryOrtho2 = cv2.imread(dryOrthoFolderO2 + '\\' + dryOrtho2)
    dryOrtho4 = [file for file in os.listdir(dryOrthoFolderO4) if not file.endswith('Thumbs.db')][0]
    dryOrtho4 = cv2.imread(dryOrthoFolderO4 + '\\' + dryOrtho4)
    dryOrtho5 = [file for file in os.listdir(dryOrthoFolderO5) if not file.endswith('Thumbs.db')][0]
    dryOrtho5 = cv2.imread(dryOrthoFolderO5 + '\\' + dryOrtho5)
    # Mask already applied to the input imagery
    dryOrtho_masked0 = dryOrtho0
    dryOrtho_masked1 = dryOrtho1
    dryOrtho_masked2 = dryOrtho2
    dryOrtho_masked4 = dryOrtho4
    dryOrtho_masked5 = dryOrtho5
    # Stack all orthos
    dryOrtho_masked = np.dstack((dryOrtho_masked0,dryOrtho_masked1,dryOrtho_masked2,dryOrtho_masked4,dryOrtho_masked5))
    input_dry_dataO = dryOrtho_masked.reshape(-1,np.shape(dryOrtho_masked)[-1])
    orthosO.append(dryOrtho_masked)
    
    # Load wet orthos
    # List and load ortho information of specific concentration
    orthosList0 = [file for file in os.listdir(orthoFolderO0) if not file.endswith('Thumbs.db')]
    orthosO0 = []
    orthosList1 = [file for file in os.listdir(orthoFolderO1) if not file.endswith('Thumbs.db')]
    orthosO1 = []
    orthosList2 = [file for file in os.listdir(orthoFolderO2) if not file.endswith('Thumbs.db')]
    orthosO2 = []
    orthosList4 = [file for file in os.listdir(orthoFolderO4) if not file.endswith('Thumbs.db')]
    orthosO4 = []
    orthosList5 = [file for file in os.listdir(orthoFolderO5) if not file.endswith('Thumbs.db')]
    orthosO5 = []
    # Loop through the orthos
    for i in range(len(orthosList0)):
        # ortho0
        ortho0 = orthosList0[i]
        # Append to ortho0ints for all val iterations
        ortho0ints.append(int(ortho0[-8:-4]))
        currentOrtho0 = cv2.imread(orthoFolderO0 + '\\' + ortho0)
        # Mask is already applied
        maskedOrtho0 = currentOrtho0
        # ortho1
        ortho1 = orthosList1[i]
        currentOrtho1 = cv2.imread(orthoFolderO1 + '\\' + ortho1)
        # Mask is already applied
        maskedOrtho1 = currentOrtho1
        # ortho2
        ortho2 = orthosList2[i]
        currentOrtho2 = cv2.imread(orthoFolderO2 + '\\' + ortho2)
        # Mask is already applied
        maskedOrtho2 = currentOrtho2
        # ortho4
        ortho4 = orthosList4[i]
        currentOrtho4 = cv2.imread(orthoFolderO4 + '\\' + ortho4)
        # Mask is already applied
        maskedOrtho4 = currentOrtho4
        # ortho5
        ortho5 = orthosList5[i]
        currentOrtho5 = cv2.imread(orthoFolderO5 + '\\' + ortho5)
        # Mask is already applied
        maskedOrtho5 = currentOrtho5
        # Remove rows and columns that have all zeroes
        maskedOrtho0 = maskedOrtho0[~np.all(maskedOrtho0[:,:,0] == 0, axis=1),:] # Rows
        maskedOrtho0 = maskedOrtho0[:,~np.all(maskedOrtho0[:,:,0] == 0, axis=0)] # Columns
        maskedOrtho1 = maskedOrtho1[~np.all(maskedOrtho1[:,:,0] == 0, axis=1),:] # Rows
        maskedOrtho1 = maskedOrtho1[:,~np.all(maskedOrtho1[:,:,0] == 0, axis=0)] # Columns
        maskedOrtho2 = maskedOrtho2[~np.all(maskedOrtho2[:,:,0] == 0, axis=1),:] # Rows
        maskedOrtho2 = maskedOrtho2[:,~np.all(maskedOrtho2[:,:,0] == 0, axis=0)] # Columns
        maskedOrtho4 = maskedOrtho4[~np.all(maskedOrtho4[:,:,0] == 0, axis=1),:] # Rows
        maskedOrtho4 = maskedOrtho4[:,~np.all(maskedOrtho4[:,:,0] == 0, axis=0)] # Columns
        maskedOrtho5 = maskedOrtho5[~np.all(maskedOrtho5[:,:,0] == 0, axis=1),:] # Rows
        maskedOrtho5 = maskedOrtho5[:,~np.all(maskedOrtho5[:,:,0] == 0, axis=0)] # Columns
        # Append with mask applied
        orthosO0.append(maskedOrtho0)
        orthosO1.append(maskedOrtho1)
        orthosO2.append(maskedOrtho2)
        orthosO4.append(maskedOrtho4)
        orthosO5.append(maskedOrtho5)
        
    del ortho0, ortho1, ortho2, ortho4, ortho5
    del orthosList0, orthosList1, orthosList2, orthosList4, orthosList5
    del currentOrtho0, currentOrtho1, currentOrtho2,currentOrtho4, currentOrtho5
    del maskedOrtho0, maskedOrtho1, maskedOrtho2, maskedOrtho4, maskedOrtho5
 
    # Combine them into single variables
    for i in range(len(orthosO0)):
        orthosO.append(np.dstack((orthosO0[i],orthosO1[i],orthosO2[i],orthosO4[i],orthosO5[i])))
    del orthosO0, orthosO1, orthosO2, orthosO4, orthosO5
    
    # Copy original data
    orthosFull = orthosO.copy()
    
    #%% First we need to stack the data in a proper way, filtering out the masked data
    # Stack orthos
    orthosStackedO = np.vstack(orthosO)
    # Stack information
    input_dataO = orthosStackedO.reshape(-1,np.shape(orthosStackedO)[-1])
    # Remove masked area (all zeros)
    del orthosO, orthosStackedO
    
    #%% Now determine the outputs, depending on concentration, but not validation set
    # First ortho is all dry, so zero
    weirDEMdepthMaps = [np.zeros((np.size(orthosFull[0],0),np.size(orthosFull[0],1),1))]
    waterLevels = np.zeros(len(ortho0ints))
    for i in range(len(ortho0ints)):
       # Store weir levels in m (/100000) and include the offset
       waterLevels[i] = ortho0ints[i]/100000 + weirOffset 
       # Make depth maps from DEM and weir levels
       weirDEMdepthMap = waterLevels[i] - zMask
       # Set negative values to zero
       weirDEMdepthMap[weirDEMdepthMap < 0] = 0
       # Convert to 3D
       weirDEMdepthMap = weirDEMdepthMap[:, :, np.newaxis]
       # Append
       weirDEMdepthMaps.append(weirDEMdepthMap)
   
    del weirDEMdepthMap, waterLevels
    
    # Copy data
    weirDEMdepthMapsFull = weirDEMdepthMaps.copy()
    
    # Reshape to three dimensions
    weirDEMdepthMapsStacked = np.vstack(weirDEMdepthMaps)
    input_dataStackedO = np.vstack(input_dataO)
    # Stack information
    output_data = weirDEMdepthMapsStacked.reshape(-1,np.shape(weirDEMdepthMapsStacked)[-1])
    input_data_1O = input_dataStackedO.reshape(-1,np.shape(input_dataStackedO)[-1])
    
    del weirDEMdepthMapsStacked, input_dataStackedO
            
    # Vertically stack the dry and wet data
    dryOrthoStackedO = np.vstack(input_dry_dataO)
    input_data1O = np.vstack([dryOrthoStackedO,input_data_1O])
    output_data1O = np.vstack([np.zeros([len(dryOrthoStackedO),1]),output_data])
    
    # Construct binaryMap for validation
    binaryMap = np.copy(output_data1O)
    binaryMap[binaryMap>0] = 1
    binaryMap = binaryMap.astype(int)
    binaryMap2 = pd.DataFrame(binaryMap)
    
    # What information to store?
    recalls = []
    precisions = []
    F1scores = []
    peirceSkillScores = []
    rmse = []
    bias = []
    bias2prmse = []
    falseWetPredDepths = []
    falseDryActualDepths = []
    trueWetPredDepths = []
    trueWetActualDepths = []

    # Now loop through the cams
    for cam in range(7):
        # Append concentration to relevant data
        recalls.append('Conc' + concentration + 'Cam' + str(cam))
        precisions.append('Conc' + concentration + 'Cam' + str(cam))
        F1scores.append('Conc' + concentration + 'Cam' + str(cam))
        peirceSkillScores.append('Conc' + concentration + 'Cam' + str(cam))
        rmse.append('Conc' + concentration + 'Cam' + str(cam))
        bias.append('Conc' + concentration + 'Cam' + str(cam))
        bias2prmse.append('Conc' + concentration + 'Cam' + str(cam))
        falseWetPredDepths.append('Conc' + concentration + 'Cam' + str(cam))
        falseDryActualDepths.append('Conc' + concentration + 'Cam' + str(cam))
        trueWetPredDepths.append('Conc' + concentration + 'Cam' + str(cam))
        trueWetActualDepths.append('Conc' + concentration + 'Cam' + str(cam))

        # Copy orthosFull and weirDEMdepthMaps into the desired datasets
        orthos = orthosFull.copy()
        weirDEMdepthMaps = weirDEMdepthMapsFull.copy()
        camData = orthosFull.copy()
        camDEMdepthMaps = weirDEMdepthMapsFull.copy()
        
        # Now apply the relevant cam and buffer masks
        for o in range(len(orthos)):
            # RF model input (exluding cam and buffer) for training
            orthos[o] = orthos[o]*(masksC[cam]*masksB[cam])
            weirDEMdepthMaps[o] = weirDEMdepthMaps[o]*(masksC[cam]*masksB[cam])
            # Only cam for cross-validation
            camData[o] = camData[o]*(masksCi[cam])
            camDEMdepthMaps[o] = camDEMdepthMaps[o]*(masksCi[cam])
            
            # Clear columns with all NaNs
            orthos[o] = orthos[o][:,~np.isnan(orthos[o]).all(axis=0)[:,0],:] 
            weirDEMdepthMaps[o] = weirDEMdepthMaps[o][:,~np.isnan(weirDEMdepthMaps[o]).all(axis=0)[:,0],:] 
            camData[o] = camData[o][:,~np.isnan(camData[o]).all(axis=0)[:,0],:] 
            camDEMdepthMaps[o] = camDEMdepthMaps[o][:,~np.isnan(camDEMdepthMaps[o]).all(axis=0)[:,0],:] 
        
        # Stack orthos
        orthosStacked = np.vstack(orthos)
        # Stack information
        input_data = orthosStacked.reshape(-1,np.shape(orthosStacked)[-1])
        
        # Continue with the weirDEMdepthMaps
        # Reshape to three dimensions
        weirDEMdepthMapsStacked = np.vstack(weirDEMdepthMaps)
        # Stack information
        output_data = weirDEMdepthMapsStacked.reshape(-1,np.shape(weirDEMdepthMapsStacked)[-1])
        # Create new binary map
        binaryMapF = np.copy(output_data)
        binaryMapF[binaryMapF>0] = 1
        binaryMapF = binaryMapF.astype(int)
        binaryMapF2 = pd.DataFrame(binaryMapF)
        
        # Make output data 1-dimensional for plotting
        output_data_1 = output_data[:,0]
        
        del orthos, orthosStacked, weirDEMdepthMaps, weirDEMdepthMapsStacked
        
        # Repeat these steps for the validation data
        # Only cam
        camDataStacked = np.vstack(camData)
        camData_data = camDataStacked.reshape(-1,np.shape(camDataStacked)[-1])
        camDEMdepthMapsStacked = np.vstack(camDEMdepthMaps)
        camoutput_data = camDEMdepthMapsStacked.reshape(-1,np.shape(camDEMdepthMapsStacked)[-1])
        camoutput_data = camoutput_data[:,0]
        binaryMapC = np.copy(camoutput_data)
        binaryMapC[binaryMapC>0] = 1
        binaryMapC = binaryMapC.astype(int)
        binaryMapC2 = pd.DataFrame(binaryMapC)
        del camData, camDataStacked, camDEMdepthMaps, camDEMdepthMapsStacked
        
        binaryMaps = [binaryMapF2,binaryMapC2]
        outputs = [output_data_1,camoutput_data]
        
        print("All data has been prepared for random forest CV model construction for concentration " + concentration + " and cam" + str(cam+1))

        #%% Now construct the random forest classifier model
        clfFile = clfFolder + '\\rFClfCVcam' + str(cam+1) + 'Conc' + concentration + '.joblib'
        
        # Only if it doesn't already exist
        if not os.path.exists(clfFile):
            print('Computing Clf model for Conc' + concentration + " and cam" + str(cam+1))
            dtnow = datetime.datetime.now()
            clf = RandomForestClassifier(n_estimators=100,max_depth=100, random_state=42, max_features=max_feature_Clf)
            clf = clf.fit(input_data, binaryMapF2.values.ravel())
            # Store the model
            joblib.dump(clf, clfFile)
            dtnow2 = datetime.datetime.now()
            dtnow3 = round((dtnow2 - dtnow).total_seconds())
            print("Clf Model computed and stored for Conc" + concentration + " and cam" + str(cam+1) + " in " + str(datetime.timedelta(seconds=dtnow3)) + " seconds")
        else:
            # Load model
            print('Loading Clf model for Conc' + concentration + " and cam" + str(cam+1))
            dtnow = datetime.datetime.now()
            clf = joblib.load(clfFile)
            dtnow2 = datetime.datetime.now()
            dtnow3 = round((dtnow2 - dtnow).total_seconds())
            print("Clf model loaded for Conc" + concentration + " and cam" + str(cam+1) + " in " + str(datetime.timedelta(seconds=dtnow3)) + " seconds")
        
        #%% Construct the regressor model
        regrFile = regrFolder + '\\rFRegrCVcam' + str(cam+1) + 'Conc' + concentration + '.joblib'
        
        # Only if it doesn't already exist
        if not os.path.exists(regrFile):
            print('Computing Regr model for Conc' + concentration + " and cam" + str(cam+1))
            dtnow = datetime.datetime.now()
            regr = RandomForestRegressor(n_estimators=100,max_depth=100, random_state=42, max_features=max_feature_Regr)
            regr.fit(input_data,output_data)
            # Store the model
            joblib.dump(regr, regrFile)
            dtnow2 = datetime.datetime.now()
            dtnow3 = round((dtnow2 - dtnow).total_seconds())
            print("Regr Model computed and stored for Conc" + concentration + " and cam" + str(cam+1) + " in " + str(datetime.timedelta(seconds=dtnow3)) + " seconds")
        else:
            # Load model
            print('Loading Regr model for Conc' + concentration + " and cam" + str(cam+1))
            dtnow = datetime.datetime.now()
            regr = joblib.load(regrFile)
            dtnow2 = datetime.datetime.now()
            dtnow3 = round((dtnow2 - dtnow).total_seconds())
            print("Regr model loaded for Conc" + concentration + " and cam" + str(cam+1) + " in " + str(datetime.timedelta(seconds=dtnow3)) + " seconds")
        
        camData_dataV = camData_data.copy()
        camoutput_dataV = camoutput_data.copy()

        print("All data is prepared for validation")
                
        #%% Conduct all validations on classifier model
        dtnow = datetime.datetime.now()
        # Cam
        validationCamClf = clf.predict(camData_dataV)
        
        # We're also interested in the indices of false positives and false negatives
        falseWetClf = np.where(np.logical_and(validationCamClf == 1,binaryMapC == 0))[0]
        falseDryClf = np.where(np.logical_and(validationCamClf == 0,binaryMapC == 1))[0]
        # And for the comparison between predicted true depths at actual depths
        trueWetClf = np.where(np.logical_and(validationCamClf == 1,binaryMapC == 1))[0]
        
        # Calculate relevant scores of the clf model
        recalls.append(metrics.recall_score(binaryMapC,validationCamClf,average=None))
        precisions.append(metrics.precision_score(binaryMapC,validationCamClf,average=None))
        F1scores.append(metrics.f1_score(binaryMapC,validationCamClf,average=None))
    
        # We're also interested in Peirce skill score, for which we use a different package
        xrclfResultWet = xr.DataArray(validationCamClf)
        xrbinaryMapWet = xr.DataArray(binaryMapC2)
        
        contingency_manager_wet = scores.categorical.BinaryContingencyManager(xrclfResultWet, xrbinaryMapWet)
        peirceSkillScores.append(contingency_manager_wet.peirce_skill_score().values) # Peirce skill score is equal for dry and wet
        
        dtnow2 = datetime.datetime.now()
        dtnow3 = round((dtnow2 - dtnow).total_seconds())
        print("Clf model validated in " + str(datetime.timedelta(seconds=dtnow3)) + " seconds")
        
        #%% Conduct all validations on regressor model
        dtnow = datetime.datetime.now()
        # Cam
        validationCamRegr = regr.predict(camData_dataV)
        
        # Combine results
        combResult = validationCamRegr * validationCamClf
        combResultWet = combResult[validationCamClf==1]
        waterDepthsWet = camoutput_data[validationCamClf==1]
        
        # RMSE
        rmse.append(np.sqrt(metrics.mean_squared_error(waterDepthsWet, combResultWet)))
        
        # Bias # == mean error
        bias.append(-1*np.mean(waterDepthsWet - combResultWet))
        
        # Coefficient bias2 as fraction of rmse
        bias2prmse.append((bias[-1]**2)/(rmse[-1]**2))
        
        # Now store the predicted depths of false wet cells and true depths of false dry cells
        falseWetPredDepths.append(combResult[falseWetClf])
        falseDryActualDepths.append(camoutput_data[falseDryClf])
        
        # As well as the true depths (for heatmaps)
        trueWetPredDepths.append(combResult[trueWetClf])
        trueWetActualDepths.append(camoutput_data[trueWetClf])
        
        dtnow2 = datetime.datetime.now()
        dtnow3 = round((dtnow2 - dtnow).total_seconds())
        print("Regr model validated in " + str(datetime.timedelta(seconds=dtnow3)) + " seconds")

    #%% Write all information to CSV files
    # Single variable data of training datasets
    with open(valFolder + '\\CV_Conc' + concentration + '_validationData.csv', 'w', newline='') as csvFile0:
        csvFile0 = csv.writer(csvFile0,delimiter=",")
        csvFile0.writerow(['Model_Conc','F1Wet','F1Dry','Peirce','RecallWet','RecallDry',
                           'PrecisionWet','PrecisionDry','RMSE','Bias','Bias2FracMSE'])
        for i in range(0,len(F1scores),2):
            csvFile0.writerow([F1scores[i],F1scores[i+1][0],F1scores[i+1][1],np.atleast_1d(peirceSkillScores[i+1])[0],
                               recalls[i+1][0],recalls[i+1][1],precisions[i+1][0],precisions[i+1][1],
                               rmse[i+1],bias[i+1],bias2prmse[i+1]])
            
    # Predicted water depths from falsely predicted wet cells
    with open(valFolder + '\\CV_Conc' + concentration + '_falseWetPredDepths.csv', 'w', newline='') as csvFile1:
        csvFile1 = csv.writer(csvFile1,delimiter=",")
        for i in range(0,len(falseWetPredDepths),2):
            csvFile1.writerow([str(falseWetPredDepths[i])])
            csvFile1.writerow(falseWetPredDepths[i+1])
            
    # True water depths from falsely predicted dry cells
    with open(valFolder + '\\CV_Conc' + concentration + '_falseDryTrueDepths.csv', 'w', newline='') as csvFile2:
        csvFile2 = csv.writer(csvFile2,delimiter=",")
        for i in range(0,len(falseDryActualDepths),2):
            csvFile2.writerow([str(falseDryActualDepths[i])])
            csvFile2.writerow(falseDryActualDepths[i+1])

    # Predicted water depths from correctly predicted wet cells
    with open(valFolder + '\\CV_Conc' + concentration + '_TrueWetPredDepths.csv', 'w', newline='') as csvFile3:
        csvFile3 = csv.writer(csvFile3,delimiter=",")
        for i in range(0,len(trueWetPredDepths),2):
            csvFile3.writerow([str(trueWetPredDepths[i])])
            csvFile3.writerow(trueWetPredDepths[i+1])

    # True water depths from correctly predicted dry cells
    with open(valFolder + '\\CV_Conc' + concentration + '_TrueWetTrueDepths.csv', 'w', newline='') as csvFile4:
        csvFile4 = csv.writer(csvFile4,delimiter=",")
        for i in range(0,len(trueWetActualDepths),2):
            csvFile4.writerow([str(trueWetActualDepths[i])])
            csvFile4.writerow(trueWetActualDepths[i+1])
        
#%% How long did the run take?
end = datetime.datetime.now()
td = round((end - start).total_seconds())
print("Script running time = " + str(datetime.timedelta(seconds=td)))
