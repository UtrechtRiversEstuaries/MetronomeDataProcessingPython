'''
Code to validate the original Classifier and Regression Random Forest Models from the overhead orthomaics
on the camera exclusive zones
The input is the same Burst imagery as for the validation datasets of the original Random Forest models
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
target_names = ['Dry','Wet']

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

#%% Determine relevant directories
demFolder = pwd + r'\\dem\\'
maskFolder = pwd + r'\\mask\\'
modelFolder = pwd + r'\\models\\'
camMaskFolder = pwd + r'\\camMasks\\'
validationFolder = pwd + r'\\validation\\Original_on_Burst_exclusive_camZones\\'
expValFolder = pwd + r'\\smGv\\'

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
dryOrthoFolder0_0 = expValFolder + r'\\RGB\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\Dry'
dryOrthoFolder1_0 = expValFolder + r'\\LAB\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\Dry'
dryOrthoFolder2_0 = expValFolder + r'\\Luv\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\Dry'
dryOrthoFolder4_0 = expValFolder + r'\\YCrCb\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\Dry'
dryOrthoFolder5_0 = expValFolder + r'\\HSV\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\Dry'

#%% Loop over concentrations
for c in range(len(concentrations)):
    concentration = concentrations[c]
    max_feature_Clf = max_features_Clf[c]
    max_feature_Regr = max_features_Regr[c]
    print('Computing validations for concentration ' + concentration)
    
    # Determine model directories
    clfFolder = modelFolder + r"\\Conc" + concentration + "\\randomForestDryWetClassifierSmG\\Settings_" + str(smGsettings[0]) + "_" + str(smGsettings[1]) + "_" + str(smGsettings[2]) + "_" +  str(smGsettings[3]) + "_mF" + str(max_feature_Clf)
    regrFolder = modelFolder + r"\\Conc" + concentration + "\\randomForestRegressionSmG\\Settings_" + str(smGsettings[0]) + "_" + str(smGsettings[1]) + "_" + str(smGsettings[2]) + "_" +  str(smGsettings[3]) + "_mF" + str(max_feature_Regr)
    valFolder = validationFolder + r"\\validationsActualModelsBurst\\Conc" + concentration
    
    # Create folder if it doesn't exist
    if not os.path.exists(valFolder):
        os.makedirs(valFolder)
    
    # Load the current models
    clfFile = clfFolder + '\\rFClfMF' + str(max_feature_Clf) + 'Conc' + concentration + 'Grid' + str(orthoRes) + 'mm.joblib'
    clf = joblib.load(clfFile)
    print("Classifier model loaded for Conc" + concentration)
    regrFile = regrFolder + '\\rFRegrMF' + str(max_feature_Regr) + 'Conc' + concentration + 'Grid' + str(orthoRes) + 'mm.joblib'
    regr = joblib.load(regrFile)
    print("Regression model loaded for Conc" + concentration)
    
    # Determine root directories from the validation wet orthos
    orthoFolder0_0 = expValFolder + '\\RGB\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\Conc' + concentration
    orthoFolder1_0 = expValFolder + '\\LAB\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\Conc' + concentration
    orthoFolder2_0 = expValFolder + '\\Luv\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\Conc' + concentration
    orthoFolder4_0 = expValFolder + '\\YCrCb\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\Conc' + concentration
    orthoFolder5_0 = expValFolder + '\\HSV\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\Conc' + concentration
    
    #%% Determine the amount of validation sets
    val_folders0 = os.listdir(orthoFolder0_0)
    val_folders = []
    for v0 in val_folders0:
         val_folders.append(v0)
    
    # Temporary data to store
    ortho0ints = []
    input_data_orthos = []
    binary_map_data = []
    waterDepth_data = []
    input_data_orthos_temp = []
    binary_map_data_temp = []
    waterDepth_data_temp = []
    orthos = []
    weirDEMdepthMaps = []
    
    # Loop through val folders
    for val in val_folders:
        # Filter out thumbs files
        if 'Thumbs' not in val:
            input_dry_data = []
            # Update root directories
            orthoFolder0 = orthoFolder0_0 + '\\' + val
            orthoFolder1 = orthoFolder1_0 + '\\' + val
            orthoFolder2 = orthoFolder2_0 + '\\' + val
            orthoFolder4 = orthoFolder4_0 + '\\' + val
            orthoFolder5 = orthoFolder5_0 + '\\' + val
            dryOrthoFolder0 = dryOrthoFolder0_0 + '\\' + val
            dryOrthoFolder1 = dryOrthoFolder1_0 + '\\' + val
            dryOrthoFolder2 = dryOrthoFolder2_0 + '\\' + val
            dryOrthoFolder4 = dryOrthoFolder4_0 + '\\' + val
            dryOrthoFolder5 = dryOrthoFolder5_0 + '\\' + val
        
            # Mask is already applied to all training and validation data
            # Load dry orthos
            dryOrtho0 = [file for file in os.listdir(dryOrthoFolder0) if not file.endswith('Thumbs.db')][0]
            dryOrtho0 = cv2.imread(dryOrthoFolder0 + '\\' + dryOrtho0)
            dryOrtho1 = [file for file in os.listdir(dryOrthoFolder1) if not file.endswith('Thumbs.db')][0]
            dryOrtho1 = cv2.imread(dryOrthoFolder1 + '\\' + dryOrtho1)
            dryOrtho2 = [file for file in os.listdir(dryOrthoFolder2) if not file.endswith('Thumbs.db')][0]
            dryOrtho2 = cv2.imread(dryOrthoFolder2 + '\\' + dryOrtho2)
            dryOrtho4 = [file for file in os.listdir(dryOrthoFolder4) if not file.endswith('Thumbs.db')][0]
            dryOrtho4 = cv2.imread(dryOrthoFolder4 + '\\' + dryOrtho4)
            dryOrtho5 = [file for file in os.listdir(dryOrthoFolder5) if not file.endswith('Thumbs.db')][0]
            dryOrtho5 = cv2.imread(dryOrthoFolder5 + '\\' + dryOrtho5)
           
            # Stack all orthos and reshape
            dryOrthos = np.dstack((dryOrtho0,dryOrtho1,dryOrtho2,dryOrtho4,dryOrtho5))
            #input_dry_data = dryOrthos.reshape(-1,np.shape(dryOrthos)[-1])
            orthos.append(dryOrthos)

            # Load wet orthos
            # List and load ortho information of specific concentration
            orthosList0 = [file for file in os.listdir(orthoFolder0) if not file.endswith('Thumbs.db')]
            orthos0 = []
            orthosList1 = [file for file in os.listdir(orthoFolder1) if not file.endswith('Thumbs.db')]
            orthos1 = []
            orthosList2 = [file for file in os.listdir(orthoFolder2) if not file.endswith('Thumbs.db')]
            orthos2 = []
            orthosList4 = [file for file in os.listdir(orthoFolder4) if not file.endswith('Thumbs.db')]
            orthos4 = []
            orthosList5 = [file for file in os.listdir(orthoFolder5) if not file.endswith('Thumbs.db')]
            orthos5 = []
            # Loop through the orthos
            for i in range(len(orthosList0)):
                # ortho0
                ortho0 = orthosList0[i]
                # Append to ortho0ints for all val iterations
                if val == 'val1':
                    ortho0ints.append(int(ortho0[-13:-9]))
                currentOrtho0 = cv2.imread(orthoFolder0 + '\\' + ortho0)
                # ortho1
                ortho1 = orthosList1[i]
                currentOrtho1 = cv2.imread(orthoFolder1 + '\\' + ortho1)
                # ortho2
                ortho2 = orthosList2[i]
                currentOrtho2 = cv2.imread(orthoFolder2 + '\\' + ortho2)
                # ortho4
                ortho4 = orthosList4[i]
                currentOrtho4 = cv2.imread(orthoFolder4 + '\\' + ortho4)
                # ortho5
                ortho5 = orthosList5[i]
                currentOrtho5 = cv2.imread(orthoFolder5 + '\\' + ortho5)
                # Append orthos
                orthos0.append(currentOrtho0)
                orthos1.append(currentOrtho1)
                orthos2.append(currentOrtho2)
                orthos4.append(currentOrtho4)
                orthos5.append(currentOrtho5)
                
            del orthoFolder0, orthoFolder1, orthoFolder2, orthoFolder4, orthoFolder5
            del ortho0, ortho1, ortho2, ortho4, ortho5
            del orthosList0, orthosList1, orthosList2, orthosList4, orthosList5
            del currentOrtho0, currentOrtho1, currentOrtho2,currentOrtho4, currentOrtho5
         
            # Combine them into single variables
            
            for i in range(len(orthos0)):
                orthos.append(np.dstack((orthos0[i],orthos1[i],orthos2[i],orthos4[i],orthos5[i])))
            del orthos0, orthos1, orthos2, orthos4, orthos5
            
            #%% Now determine the outputs, depending on concentration, but not validation set
            # Also append empty weirDepthMap
            weirDEMdepthMaps.append(np.zeros((24,672,1)))
            
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
    
    # Copy original data
    orthosFull = orthos.copy()        
    weirDEMdepthMapsFull = weirDEMdepthMaps.copy()

    #%% What information to store?
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
        
        print("All data has been prepared for validation on concentration " + concentration + " and cam" + str(cam+1))

        camData_dataV = camData_data.copy()
        camoutput_dataV = camoutput_data.copy()

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
    with open(valFolder + '\\AB_Conc' + concentration + '_validationData.csv', 'w', newline='') as csvFile0:
        csvFile0 = csv.writer(csvFile0,delimiter=",")
        csvFile0.writerow(['Model_Conc','F1Wet','F1Dry','Peirce','RecallWet','RecallDry',
                           'PrecisionWet','PrecisionDry','RMSE','Bias','Bias2FracMSE'])
        for i in range(0,len(F1scores),2):
            csvFile0.writerow([F1scores[i],F1scores[i+1][0],F1scores[i+1][1],np.atleast_1d(peirceSkillScores[i+1])[0],
                               recalls[i+1][0],recalls[i+1][1],precisions[i+1][0],precisions[i+1][1],
                               rmse[i+1],bias[i+1],bias2prmse[i+1]])
            
    # Predicted water depths from falsely predicted wet cells
    with open(valFolder + '\\AB_Conc' + concentration + '_falseWetPredDepths.csv', 'w', newline='') as csvFile1:
        csvFile1 = csv.writer(csvFile1,delimiter=",")
        for i in range(0,len(falseWetPredDepths),2):
            csvFile1.writerow([str(falseWetPredDepths[i])])
            csvFile1.writerow(falseWetPredDepths[i+1])
            
    # True water depths from falsely predicted dry cells
    with open(valFolder + '\\AB_Conc' + concentration + '_falseDryTrueDepths.csv', 'w', newline='') as csvFile2:
        csvFile2 = csv.writer(csvFile2,delimiter=",")
        for i in range(0,len(falseDryActualDepths),2):
            csvFile2.writerow([str(falseDryActualDepths[i])])
            csvFile2.writerow(falseDryActualDepths[i+1])

    # Predicted water depths from correctly predicted wet cells
    with open(valFolder + '\\AB_Conc' + concentration + '_TrueWetPredDepths.csv', 'w', newline='') as csvFile3:
        csvFile3 = csv.writer(csvFile3,delimiter=",")
        for i in range(0,len(trueWetPredDepths),2):
            csvFile3.writerow([str(trueWetPredDepths[i])])
            csvFile3.writerow(trueWetPredDepths[i+1])

    # True water depths from correctly predicted dry cells
    with open(valFolder + '\\AB_Conc' + concentration + '_TrueWetTrueDepths.csv', 'w', newline='') as csvFile4:
        csvFile4 = csv.writer(csvFile4,delimiter=",")
        for i in range(0,len(trueWetActualDepths),2):
            csvFile4.writerow([str(trueWetActualDepths[i])])
            csvFile4.writerow(trueWetActualDepths[i+1])
        
#%% How long did the run take?
end = datetime.datetime.now()
td = round((end - start).total_seconds())
print("Script running time = " + str(datetime.timedelta(seconds=td)))
