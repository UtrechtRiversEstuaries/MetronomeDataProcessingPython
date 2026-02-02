'''
Code to apply cross-model validation of the random forest models and extract all the relevant information.
The validations in this script are applicable for both training and validation datasets.

Python script written by Eise W. Nota (finalized JANUARY 2026)

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
models = ['0','1','2','3','4','5']
whichMask = '2' # either '1' (excluding  actuators) or '2' (including actuators)
sampleSize = None # What is the max sample size from which the random forest is calculated? Set None if you want to include all samples. Note: the windows server can't cope with all samples of the finest grid sizes. 
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
differenceWaterLevel = 0.00 # Maybe 0 because of capillary forces
demfloodLevel = 0.102
weirOffset = round((demfloodLevel - differenceWaterLevel) - finalWeirLevel,5)

#%% Determine relevant directories
demFolder = pwd + r'\\dem\\'
maskFolder = pwd + r'\\mask\\'
writeFolder = pwd + r'\\validationData\\'
expCalibFolder = pwd +  r'\\smG\\'
expValFolder = pwd +  r'\\smGv\\'
modelFolder = pwd + r'\\models\\'

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
# Remove rows and columans that have all zeroes
zMask = zMask[~np.isnan(zMask).all(axis=1),:] # Rows
zMask = zMask[:,~np.isnan(zMask).all(axis=0)] # Columns

del(netCDFFile, zData, pdMask, DEM_name)

#%% Determine root directories from the validation dry orthos
dryOrthoFolder0_0 = expValFolder + r'\\RGB\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\Dry'
dryOrthoFolder1_0 = expValFolder + r'\\LAB\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\Dry'
dryOrthoFolder2_0 = expValFolder + r'\\Luv\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\Dry'
dryOrthoFolder4_0 = expValFolder + r'\\YCrCb\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\Dry'
dryOrthoFolder5_0 = expValFolder + r'\\HSV\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\Dry'
# As well as from the training dataset
dryOrthoFolderO0_0 = expCalibFolder + r'\\RGB\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\Dry'
dryOrthoFolderO1_0 = expCalibFolder + r'\\LAB\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\Dry'
dryOrthoFolderO2_0 = expCalibFolder + r'\\Luv\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\Dry'
dryOrthoFolderO4_0 = expCalibFolder + r'\\YCrCb\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\Dry'
dryOrthoFolderO5_0 = expCalibFolder + r'\\HSV\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\Dry'

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

#%% Loop over models 
for model in models:
    print('Computing validations for Models' + model)
    # Load the current models
    clfFile = modelFolder + '\\Conc' + model + 'Clf.joblib'
    regrFile = modelFolder + '\\Conc' + model + 'Regr.joblib'
    clf = joblib.load(clfFile)
    print("Classifier model loaded for Conc" + model)
    regr = joblib.load(regrFile)
    print("Regression model loaded for Conc" + model)
    
    #%% Loop over concentrations
    for concentration in concentrations:
        print("Computing validations for Models" + model + " and Conc" + concentration)
        
        # Append concentration to relevant data
        recalls.append('Model' + model + 'Conc' + concentration)
        precisions.append('Model' + model + 'Conc' + concentration)
        F1scores.append('Model' + model + 'Conc' + concentration)
        peirceSkillScores.append('Model' + model + 'Conc' + concentration)
        rmse.append('Model' + model + 'Conc' + concentration)
        bias.append('Model' + model + 'Conc' + concentration)
        bias2prmse.append('Model' + model + 'Conc' + concentration)
        falseWetPredDepths.append('Model' + model + 'Conc' + concentration)
        falseDryActualDepths.append('Model' + model + 'Conc' + concentration)
        trueWetPredDepths.append('Model' + model + 'Conc' + concentration)
        trueWetActualDepths.append('Model' + model + 'Conc' + concentration)
        
        # Determine root directories from the validation wet orthos
        orthoFolder0_0 = expValFolder + '\\RGB\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\Conc' + concentration
        orthoFolder1_0 = expValFolder + '\\LAB\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\Conc' + concentration
        orthoFolder2_0 = expValFolder + '\\Luv\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\Conc' + concentration
        orthoFolder4_0 = expValFolder + '\\YCrCb\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\Conc' + concentration
        orthoFolder5_0 = expValFolder + '\\HSV\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\Conc' + concentration
        # As well as from the training dataset
        orthoFolderO0_0 = expCalibFolder + '\\RGB\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\Conc' + concentration
        orthoFolderO1_0 = expCalibFolder + '\\LAB\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\Conc' + concentration
        orthoFolderO2_0 = expCalibFolder + '\\Luv\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\Conc' + concentration
        orthoFolderO4_0 = expCalibFolder + '\\YCrCb\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\Conc' + concentration
        orthoFolderO5_0 = expCalibFolder + '\\HSV\\BaseModel' + baseModel + '\\Res' + str(orthoRes) + 'mm\\Conc' + concentration

        #%% Determine the amount of validation sets
        val_folders0 = os.listdir(orthoFolder0_0)
        
        # If model and concentration equal each other, we'll exclude the training dataset
        if model == concentration:
            val_folders = []
        else: # Training datasets are just another validation
            val_folders = ['original']
        
        # Append Validation sets
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
        
        # Loop through val folders
        for val in val_folders:
            # Filter out thumbs files
            if 'Thumbs' not in val:
                input_dry_data = []
                # Update root directories
                # This depends on whether it's an original or validation set
                if val == 'original':
                    orthoFolder0 = orthoFolderO0_0
                    orthoFolder1 = orthoFolderO1_0
                    orthoFolder2 = orthoFolderO2_0
                    orthoFolder4 = orthoFolderO4_0
                    orthoFolder5 = orthoFolderO5_0
                    dryOrthoFolder0 = dryOrthoFolderO0_0
                    dryOrthoFolder1 = dryOrthoFolderO1_0 
                    dryOrthoFolder2 = dryOrthoFolderO2_0 
                    dryOrthoFolder4 = dryOrthoFolderO4_0 
                    dryOrthoFolder5 = dryOrthoFolderO5_0 
                else: # Validation sets
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
                input_dry_data = dryOrthos.reshape(-1,np.shape(dryOrthos)[-1])
                
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
                    if val == val_folders[0]:
                        # Depends on whether it's original folder of val folder
                        if val == 'original': # Training dataset
                            ortho0ints.append(int(ortho0[-8:-4]))
                        else: # Validation dataset
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
                orthos = []
                for i in range(len(orthos0)):
                    orthos.append(np.dstack((orthos0[i],orthos1[i],orthos2[i],orthos4[i],orthos5[i])))
                del orthos0, orthos1, orthos2, orthos4, orthos5
                
                
                #%% First we need to stack the data in a proper way
                # Stack orthos
                orthosStacked = np.vstack(orthos)
                # Stack information
                input_data = orthosStacked.reshape(-1,np.shape(orthosStacked)[-1])
                
                del orthos, orthosStacked
                
                #%% Now determine the outputs, depending on concentration, but not validation set
                weirDEMdepthMaps = []
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
                
                # Reshape to three dimensions
                weirDEMdepthMapsStacked = np.vstack(weirDEMdepthMaps)
                input_dataStacked = np.vstack(input_data)
                # Stack information
                output_data = weirDEMdepthMapsStacked.reshape(-1,np.shape(weirDEMdepthMapsStacked)[-1])
                input_data_1 = input_dataStacked.reshape(-1,np.shape(input_dataStacked)[-1])
                
                del weirDEMdepthMapsStacked, input_dataStacked     # , weirDEMdepthMaps
                        
                # Vertically stack the dry and wet data
                dryOrthoStacked = np.vstack(input_dry_data)
                input_data1 = np.vstack([dryOrthoStacked,input_data_1])
                output_data1 = np.vstack([np.zeros([len(dryOrthoStacked),1]),output_data])
                
                # Construct binaryMap for validation
                binaryMap = np.copy(output_data1)
                binaryMap[binaryMap>0] = 1
                binaryMap = binaryMap.astype(int)
                binaryMap2 = pd.DataFrame(binaryMap)
                
                # Append training and validation data
                # As no training datasets are used here, all sets are validation datasets
                input_data_orthos_temp.append(input_data1)
                binary_map_data_temp.append(binaryMap)
                waterDepth_data_temp.append(output_data1)
                
        #%% Vertically stack all data from the validation datasets
        input_data_orthos.append(np.vstack(input_data_orthos_temp))
        binary_map_data.append(np.vstack(binary_map_data_temp))
        waterDepth_data.append(np.vstack(waterDepth_data_temp))
        
        del input_data_orthos_temp, binary_map_data_temp, waterDepth_data_temp, binaryMap, binaryMap2, input_data1, output_data1, input_data_1
        
        print("All data has been prepared for random forest validation on Models" + model + " and Conc" + concentration)
                
        #%%  Validate the random forest for the validation datasets
        for ds in range(len(input_data_orthos)):
            currentInput = input_data_orthos[ds]
            currentBinaryMap = binary_map_data[ds] #np.hstack(binary_map_data[ds])
            currentwaterDepths = np.hstack(waterDepth_data[ds])
            
            # Apply classifier model to dataset
            clfResult = clf.predict(currentInput)
            
            # We're also interested in the indices of false positives and false negatives
            falseWetClf = np.where(np.logical_and(clfResult == 1,currentBinaryMap[:,0] == 0))[0]
            falseDryClf = np.where(np.logical_and(clfResult == 0,currentBinaryMap[:,0] == 1))[0]
            # And for the comparison between predicted true depths at actual depths
            trueWetClf = np.where(np.logical_and(clfResult == 1,currentBinaryMap[:,0] == 1))[0]
            
            # Calculate relevant scores of the clf model
            recalls.append(metrics.recall_score(currentBinaryMap,clfResult,average=None))
            precisions.append(metrics.precision_score(currentBinaryMap,clfResult,average=None))
            F1scores.append(metrics.f1_score(currentBinaryMap,clfResult,average=None))
        
            # We're also interested in Peirce skill score, for which we use a different package
            xrclfResultWet = xr.DataArray(clfResult)
            xrbinaryMapWet = xr.DataArray(currentBinaryMap)
            
            contingency_manager_wet = scores.categorical.BinaryContingencyManager(xrclfResultWet, xrbinaryMapWet)
            peirceSkillScores.append(contingency_manager_wet.peirce_skill_score().values) # Peirce skill score is equal for dry and wet
            # https://github.com/nci/scores/blob/develop/src/scores/categorical/contingency_impl.py
            
            #%% Apply regression model to dataset
            regrResult = regr.predict(currentInput)
            
            # Combine results
            combResult = regrResult * clfResult
            combResultWet = combResult[clfResult==1]
            waterDepthsWet = currentwaterDepths[clfResult==1]
            
            # RMSE
            rmse.append(np.sqrt(metrics.mean_squared_error(waterDepthsWet, combResultWet)))
            
            # Bias # == mean error
            bias.append(-1*np.mean(waterDepthsWet - combResultWet))
            
            # Coefficient bias2 as fraction of rmse
            bias2prmse.append((bias[-1]**2)/(rmse[-1]**2))
            
            # Now store the predicted depths of false wet cells and true depths of false dry cells
            falseWetPredDepths.append(combResult[falseWetClf])
            falseDryActualDepths.append(currentwaterDepths[falseDryClf])
            
            # As well as the true depths (for heatmaps)
            trueWetPredDepths.append(combResult[trueWetClf])
            trueWetActualDepths.append(currentwaterDepths[trueWetClf])
        
        print("All data has been validated for Models" + model + " and Conc" + concentration)
    
    print("All concentrations have been validated for Models" + model)
    
#%% Write all information to CSV files
# Single variable data of training datasets
with open(writeFolder + '\\Fig5_validationData.csv', 'w', newline='') as csvFile0:
    csvFile0 = csv.writer(csvFile0,delimiter=",")
    csvFile0.writerow([['Model_Conc'],['F1Wet'],['F1Dry'],['Peirce'],['RecallWet'],['RecallDry'],
                       ['PrecisionWet'],['PrecisionDry'],['RMSE'],['Bias'],['Bias2FracMSE']])
    for i in range(0,len(F1scores),2):
        # Trainig datasets
        csvFile0.writerow([[F1scores[i]],[F1scores[i+1][0]],[F1scores[i+1][1]],[np.atleast_1d(peirceSkillScores[i+1])[0]],
                           [recalls[i+1][0]],[recalls[i+1][1]],[precisions[i+1][0]],[precisions[i+1][1]],
                           [rmse[i+1]],[bias[i+1]],[bias2prmse[i+1]]])
        
# Predicted water depths from falsely predicted wet cells
with open(writeFolder + '\\Fig5_falseWetPredDepths.csv', 'w', newline='') as csvFile1:
    csvFile1 = csv.writer(csvFile1,delimiter=",")
    csvFile1.writerow([['Model_Conc'],['values']])
    for i in range(0,len(falseWetPredDepths),2):
        # Trainig datasets
        csvFile1.writerow([str(falseWetPredDepths[i])])
        csvFile1.writerow(falseWetPredDepths[i+1])
        
# True water depths from falsely predicted dry cells
with open(writeFolder + '\\Fig5_falseDryTrueDepths.csv', 'w', newline='') as csvFile2:
    csvFile2 = csv.writer(csvFile2,delimiter=",")
    csvFile2.writerow([['Model_Conc'],['values']])
    for i in range(0,len(falseDryActualDepths),2):
        # Trainig datasets
        csvFile2.writerow([str(falseDryActualDepths[i])])
        csvFile2.writerow(falseDryActualDepths[i+1])

# Predicted water depths from correctly predicted wet cells
with open(writeFolder + '\\Fig5_TrueWetPredDepths.csv', 'w', newline='') as csvFile3:
    csvFile3 = csv.writer(csvFile3,delimiter=",")
    csvFile3.writerow([['Concentration'],['Dataset'],['values']])
    for i in range(0,len(trueWetPredDepths),3):
        # Trainig datasets
        csvFile3.writerow([[str(trueWetPredDepths[i])],['Training']])
        csvFile3.writerow(trueWetPredDepths[i+1])
        
    for i in range(0,len(trueWetPredDepths),3):    
        # Validations datasets
        csvFile3.writerow([[str(trueWetPredDepths[i])],['Validation']])
        csvFile3.writerow(trueWetPredDepths[i+2])

# True water depths from correctly predicted dry cells
with open(writeFolder + '\\Fig5_TrueWetTrueDepths.csv', 'w', newline='') as csvFile4:
    csvFile4 = csv.writer(csvFile4,delimiter=",")
    csvFile4.writerow([['Concentration'],['Dataset'],['values']])
    for i in range(0,len(trueWetActualDepths),3):
        # Trainig datasets
        csvFile4.writerow([[str(trueWetActualDepths[i])],['Training']])
        csvFile4.writerow(trueWetActualDepths[i+1])
        
    for i in range(0,len(trueWetActualDepths),3):    
        # Validation datasets
        csvFile4.writerow([[str(trueWetActualDepths[i])],['Validation']])
        csvFile4.writerow(trueWetActualDepths[i+2])


#%% How long did the run take?
end = datetime.datetime.now()
td = round((end - start).total_seconds())
print("Script running time = " + str(datetime.timedelta(seconds=td)))
