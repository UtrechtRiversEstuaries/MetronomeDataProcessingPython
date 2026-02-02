'''
Code to construct Classifier and Regression Random Forest Models from the overhead orthomaics
to which mask2 and a Gaussian filter have been applied, as well as transformations to the
LAB, Luv, YCrCb and HSV color spaces.

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

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import joblib

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
whichMask = '2' # either '1' (excluding actuators) or '2' (including actuators)
# Gaussian smooth settings
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
            
target_names = ['Dry','Wet']

# Define weir offsets
finalWeirLevel = 0.08613
differenceWaterLevel = 0.00
demfloodLevel = 0.102
weirOffset = round((demfloodLevel - differenceWaterLevel) - finalWeirLevel,5)

#%% Determine mask and DEM directories
demFolder = pwd + r'\\dem\\'
maskFolder = pwd + r'\\mask\\'
modelFolder = pwd + r'\\models\\'

#%% Load DEM
DEM_name = 'Exp' + Expnr + '_' + DEMsurvey  + '_LaserScan_DEM.nc'
    
netCDFFile = nc.Dataset(demFolder + '\\' + DEM_name)
# Extract the relevant information
# xAxis = np.array(netCDFFile.get_variables_by_attributes(name='X-axis')[0])
# yAxis = np.array(netCDFFile.get_variables_by_attributes(name='Y-axis')[0])
# pcs = np.array(netCDFFile.get_variables_by_attributes(name='Z percentiles')[0])
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

#%% Loop over concentrations
for concentration in concentrations:
    # Determine write directories
    wdir = pwd + r'\\smG\\'
    orthoFolder0 = wdir + '\\RGB\\Conc' + concentration
    orthoFolder1 = wdir + '\\LAB\\Conc' + concentration
    orthoFolder2 = wdir + '\\Luv\\Conc' + concentration
    orthoFolder4 = wdir + '\\YCrCb\\Conc' + concentration
    orthoFolder5 = wdir + '\\HSV\\Conc' + concentration
    dryOrthoFolder0 = wdir + '\\RGB\\Dry'
    dryOrthoFolder1 = wdir + '\\LAB\\Dry'
    dryOrthoFolder2 = wdir + '\\Luv\\Dry'
    dryOrthoFolder4 = wdir + '\\YCrCb\\Dry'
    dryOrthoFolder5 = wdir + '\\HSV\\Dry'
    
    #%% Load orthomosaics
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
    # Mask is already applied to the input imagery
    
    #%% List and load ortho information of specific concentration
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
    # Mask is already applied to the input imagery
    
    weirDEMdepthMaps = []
    waterLevels = np.zeros(len(orthosList0)) # Both should have the same length
    for i in range(len(orthosList0)):
        # ortho0
        ortho0 = orthosList0[i]
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
        # Remove rows and columns that have all zeroes
        currentOrtho0 = currentOrtho0[~np.all(currentOrtho0[:,:,0] == 0, axis=1),:] # Rows
        currentOrtho0 = currentOrtho0[:,~np.all(currentOrtho0[:,:,0] == 0, axis=0)] # Columns
        currentOrtho1 = currentOrtho1[~np.all(currentOrtho1[:,:,0] == 0, axis=1),:] # Rows
        currentOrtho1 = currentOrtho1[:,~np.all(currentOrtho1[:,:,0] == 0, axis=0)] # Columns
        currentOrtho2 = currentOrtho2[~np.all(currentOrtho2[:,:,0] == 0, axis=1),:] # Rows
        currentOrtho2 = currentOrtho2[:,~np.all(currentOrtho2[:,:,0] == 0, axis=0)] # Columns
        currentOrtho4 = currentOrtho4[~np.all(currentOrtho4[:,:,0] == 0, axis=1),:] # Rows
        currentOrtho4 = currentOrtho4[:,~np.all(currentOrtho4[:,:,0] == 0, axis=0)] # Columns
        currentOrtho5 = currentOrtho5[~np.all(currentOrtho5[:,:,0] == 0, axis=1),:] # Rows
        currentOrtho5 = currentOrtho5[:,~np.all(currentOrtho5[:,:,0] == 0, axis=0)] # Columns
        # Append with mask applied
        orthos0.append(currentOrtho0)
        orthos1.append(currentOrtho1)
        orthos2.append(currentOrtho2)
        orthos4.append(currentOrtho4)
        orthos5.append(currentOrtho5)
        # Store weir levels in m (/100000) and include the offset
        waterLevels[i] = int(ortho0[-8:-4])/100000 + weirOffset 
        # Make depth maps from DEM and weir levels
        weirDEMdepthMap = waterLevels[i] - zMask
        # Set negative values to zero
        weirDEMdepthMap[weirDEMdepthMap < 0] = 0
        # Convert to 3D
        weirDEMdepthMap = weirDEMdepthMap[:, :, np.newaxis]
        # Append
        weirDEMdepthMaps.append(weirDEMdepthMap)
       
    # Stack all orthos
    dryOrtho_masked = np.dstack((dryOrtho0,dryOrtho1,dryOrtho2,dryOrtho4,dryOrtho5))
    # Continue with wet orthos
    orthos = []
    for i in range(len(orthos0)):
        orthos.append(np.dstack((orthos0[i],orthos1[i],orthos2[i],orthos4[i],orthos5[i])))
    
    del dryOrtho0, dryOrtho1, dryOrtho2, dryOrtho4, dryOrtho5
    del orthos0, orthos1, orthos2, orthos4, orthos5
    del orthosList0, orthosList1, orthosList2, orthosList4, orthosList5
    del currentOrtho0, currentOrtho1, currentOrtho2, currentOrtho4, currentOrtho5

    #%% In case more than one spectral dataset is used, we need to combine them into single variables
    # Stack information
    dryOrthoStacked = dryOrtho_masked.reshape(-1,np.shape(dryOrtho_masked)[-1])
    # Continue with the wet orthos
    # Stack orthos
    orthosStacked = np.vstack(orthos)
    # Stack information
    input_data = orthosStacked.reshape(-1,np.shape(orthosStacked)[-1])
    
    # Continue with the weirDEMdepthMaps
    # Reshape to three dimensions
    weirDEMdepthMapsStacked = np.vstack(weirDEMdepthMaps)
    # Stack information
    output_data = weirDEMdepthMapsStacked.reshape(-1,np.shape(weirDEMdepthMapsStacked)[-1])
    # Remove masked area (all zeros) # Validate this later for mask1
    output_data = output_data[~np.isnan(output_data).all(axis=1)]

    # Vertically stack the dry and wet data
    input_data1 = np.vstack([dryOrthoStacked,input_data])
    output_data1 = np.vstack([np.zeros([len(dryOrthoStacked),1]),output_data])
    
    # Make output data 1-dimensional for plotting
    output_data_1 = output_data1[:,0]
        
    del orthos, orthosStacked, weirDEMdepthMaps, weirDEMdepthMapsStacked, dryOrthoStacked
    
    #%% Now first define a binary classification model
    # We have to identify a binary map first
    # i.e. which cells are dry (z = 0) and which are wet (z > 0)
    # This depends on the validateExistingModel setting
    binaryMap = np.copy(output_data1)
    input_data2 = pd.DataFrame(input_data1)
    binaryMap[binaryMap>0] = 1
    binaryMap = binaryMap.astype(int)
    binaryMap2 = pd.DataFrame(binaryMap)

    print("All data has been prepared for random forest training")
    
    #%% Now construct the random forest classifier model
    clfFile = modelFolder + '\\rFClfConc' + concentration + '.joblib'
    
    # Only if it doesn't already exist
    if not os.path.exists(clfFile):
        print('Computing classifier model for Conc' + concentration)
        dtnow = datetime.datetime.now()
        clf = RandomForestClassifier(n_estimators=100, max_depth=100, random_state = 42, max_features = None)
        # Note: max_features have been calibrated in the final models
        clf = clf.fit(input_data2.values, binaryMap2.values.ravel())
        # Store the model
        joblib.dump(clf, clfFile)
        dtnow2 = datetime.datetime.now()
        dtnow3 = round((dtnow2 - dtnow).total_seconds())
        print("Model computed and stored for Conc" + concentration + " in " + str(datetime.timedelta(seconds=dtnow3)) + " seconds")
    else:
        # Load model
        print('Loading classifier model for Conc' + concentration)
        dtnow = datetime.datetime.now()
        # Load model
        clf = joblib.load(clfFile)
        dtnow2 = datetime.datetime.now()
        dtnow3 = round((dtnow2 - dtnow).total_seconds())
        print("Existing classifier model loaded for Conc" + concentration + " in " + str(datetime.timedelta(seconds=dtnow3)) + " seconds")
    
    #%% Construct the random forest regression model
    regrFile = modelFolder + '\\rFRegrConc' + concentration + '.joblib'
    
    # Only if it doesn't already exist
    if not os.path.exists(regrFile):
        print('Computing regressor model for Conc' + concentration)
        dtnow = datetime.datetime.now()
        regr = RandomForestRegressor(n_estimators=100, max_depth=100, random_state = 42, max_features = None)
        # Note: max_features have been calibrated in the final models
        regr = regr.fit(input_data,output_data)
        # Store the model
        joblib.dump(regr, regrFile)
        dtnow2 = datetime.datetime.now()
        dtnow3 = round((dtnow2 - dtnow).total_seconds())
        print("Model computed and stored for Conc" + concentration + " in " + str(datetime.timedelta(seconds=dtnow3)) + " seconds")
    else:
        # Load model
        print('Loading regressor model for Conc' + concentration)
        dtnow = datetime.datetime.now()
        # Load model
        regr = joblib.load(regrFile)
        dtnow2 = datetime.datetime.now()
        dtnow3 = round((dtnow2 - dtnow).total_seconds())
        print("Existing classifier model loaded for Conc" + concentration + " in " + str(datetime.timedelta(seconds=dtnow3)) + " seconds")
    
    #%% Now extract information from the models
    # All options are documented at sklearn
    featuresClf = clf.feature_importances_ # Weighted to 1
    featuresRegr = regr.feature_importances_ # Weighted to 1
    scoreClf = clf.score(input_data1,binaryMap) # R score
    scoreRegr = regr.score(input_data,output_data) # R score

    # Store feature importances
    # Store text_representation in csv file
    with open(modelFolder + '\\featureImportanceClf.csv', 'w', newline='') as featureImportanceFileClf:
        featureImportanceClfWriter = csv.writer(featureImportanceFileClf,delimiter=",")
        for feature in range(len(input_data_names)):
            featureImportanceClfWriter.writerow([[input_data_names[feature]],[str(featuresClf[feature])]])

    with open(modelFolder + '\\featureImportanceRegr.csv', 'w', newline='') as featureImportanceFileRegr:
        featureImportanceRegrWriter = csv.writer(featureImportanceFileRegr,delimiter=",")
        for feature in range(len(input_data_names)):
            featureImportanceRegrWriter.writerow([[input_data_names[feature]],[str(featuresRegr[feature])]])

#%% How long did the run take?
end = datetime.datetime.now()
td = round((end - start).total_seconds())
print("Script running time = " + str(datetime.timedelta(seconds=td)))
