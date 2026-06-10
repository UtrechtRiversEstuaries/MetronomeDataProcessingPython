'''
Code to validate the original Classifier and Regression Random Forest Models from the overhead orthomaics
AND the Source Domain-directed CV models from the overhead orthomaics on the camera exclusive zones
The input is the same Experimental imagery as for the Experimental dataset (Exp062) of the original Random Forest models
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

Exp062_orthos = [Exp062_01000_6900,Exp062_03000_6900,Exp062_09000_4636,Exp062_09000_7570,Exp062_09000_8327]
Exp062_dems = ['01000','03000','09000','09000','09000']
Exp062_weirs = ['6900','6900','4636','7570','8327']

# Relevant masks
mask062 = '062'
maskCycle062 = '00000'
maskBuffer062 = '2'

# Pre-processing method for ortho
methodO = 'smoothGaus'
borderTypeO = cv2.BORDER_ISOLATED # Default borderType
smGsettings = [5,5,0,0]

# Models to test
concentrations = ['0','1','2','3','4','5']
forestSize = 100 # How many trees in the random forest?
treeDepth = 100 # What is the maximum tree depth?
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

#%% Determine relevant directories and files
demExpFolder = pwd + r'\\demExp\\'
maskExpFolder = pwd + r'\\maskExp\\'
modelFolder = pwd + r'\\models\\'
modelCVFolder = pwd + r'\\modelsCV\\'
camMaskFolder = pwd + r'\\camMasks\\'
validationFolder = pwd + r'\\validation\\Original_on_Experimental\\'
validationCVFolder = pwd + r'\\validation\\Source_Domain-Directed_CV_on_Experimental\\'
expExpFolder = pwd + r'\\smGe\\'

# Load masks
if int(maskBuffer062) > 0:
    maskFile062 = 'MaskExp' + mask062 + 'Cycle' + maskCycle062 + 'Buffer' + maskBuffer062 + '.nc'
else:
    maskFile062 = 'MaskExp' + mask062 + 'Cycle' + maskCycle062 + '.nc'

# DEM files
demFiles062 = []
for i in range(len(Exp062_dems)):
    demFiles062.append('Exp062_' + Exp062_dems[i] + '_Laserscan_DEM.nc')

#%% Load and prepare masks
# Exp062
maskNC062 = nc.Dataset(maskExpFolder + '\\' + maskFile062, 'r')
# Extract the relevant information
mask062 = np.array(maskNC062.get_variables_by_attributes(name='mask')[0])
# Extract the relevant information
xAxis = np.array(maskNC062.get_variables_by_attributes(name='X-axis')[0])
yAxis = np.array(maskNC062.get_variables_by_attributes(name='Y-axis')[0])
# We can close the file again
maskNC062.close()
# Flip vertically for ortho
mask062 = np.flipud(mask062)

# Load cam masks
# Buffer masks are not necessary as we will only validate on cams
maskC1 = np.array(pd.read_csv(camMaskFolder + '\\maskCam1.csv', delimiter=',', header=None))
maskC2 = np.array(pd.read_csv(camMaskFolder + '\\maskCam2.csv', delimiter=',', header=None))
maskC3 = np.array(pd.read_csv(camMaskFolder + '\\maskCam3.csv', delimiter=',', header=None))
maskC4 = np.array(pd.read_csv(camMaskFolder + '\\maskCam4.csv', delimiter=',', header=None)) 
maskC5 = np.array(pd.read_csv(camMaskFolder + '\\maskCam5.csv', delimiter=',', header=None)) 
maskC6 = np.array(pd.read_csv(camMaskFolder + '\\maskCam6.csv', delimiter=',', header=None)) 

# Stack
masksC062 = [maskC1,maskC2,maskC3,maskC4,maskC5,maskC6]
del(maskC1,maskC2,maskC3,maskC4,maskC5,maskC6)

# Also make masks suited for 3D
orthoMask3062 = [[],[],[],[],[],[]]

# Loop through masksC
for i in range(len(masksC062)):
    # Set ones to NaN and zeros to ones
    masksC062[i] = masksC062[i].astype('float')
    # Multiply with mask062
    masksC062[i] = masksC062[i]*mask062
    masksC062[i][masksC062[i] == 0] = np.nan
    # Make mask suited for 3D
    orthoMask3062[i] = np.zeros([np.size(masksC062[i],0),np.size(masksC062[i],1),3])
    orthoMask3062[i][:,:,0] = masksC062[i]; orthoMask3062[i][:,:,1] = masksC062[i]; orthoMask3062[i][:,:,2] = masksC062[i]

# Define additional mask for Exp062_09000_4636 due to the upstream section not being fully drained when imagery was taken
mask062_4636 = mask062.copy()
mask062_4636[:,:260] = 0
mask062_4636[70:,:400] = 0
mask062_4636[80:,:520] = 0
mask062_4636[90:,:560] = 0

# Make nanmask
mask062_4636 = mask062_4636.astype('float')
mask062_4636[mask062_4636 == 0] = np.nan

# Make mask suited for 3D
orthoMask3062_4636 = np.zeros([np.size(mask062_4636,0),np.size(mask062_4636,1),3])
orthoMask3062_4636[:,:,0] = mask062_4636; orthoMask3062_4636[:,:,1] = mask062_4636; orthoMask3062_4636[:,:,2] = mask062_4636

#%% Load Dems and construct weirDEMDepthMaps 
# Exp062
weirDems062 = [[],[],[],[],[],[]]
weirDems062_1 = [[],[],[],[],[],[]]
weirDems062_2 = [[],[],[],[],[],[]] # Stacked
dryWetMaps062 = [[],[],[],[],[],[]]

for i in range(len(demFiles062)):
    # Current DEM
    demFile = demFiles062[i]
    # Load DEM
    netCDFFile = nc.Dataset(demExpFolder + '\\' + demFile)
    # Extract the relevant information
    zData = np.array(netCDFFile.get_variables_by_attributes(name='Z-axis')[0])
    # We can close the file again
    netCDFFile.close()
    # Make zData two-dimensional
    zData = zData[:,:,0].reshape((zData.shape[0],zData.shape[1]))
    # Mask is different for '4636'
    if Exp062_weirs[i] == '4636':
        zData = zData*mask062_4636
    # Make depth maps from DEM and weir levels
    waterLevel = float(Exp062_weirs[i])/100000 + weirOffset
    
    #%% Now loop per camera
    for j in range(len(weirDems062)):
        # Combine zData with cammask
        # Apply a vertical flip due to inversed y-axes between mask and zData
        zMask062 = zData*np.flip(masksC062[j],axis=0)
        
        # Construct weirDEMDepthMap
        weirDEMdepthMap062 = waterLevel - zMask062
        # Set negative values to zero
        weirDEMdepthMap062[weirDEMdepthMap062 < 0] = 0
    
        # Construct boolean dry/wet map from weirDEMdepthMap
        weirDEMDryWetMap062 = weirDEMdepthMap062.copy()
        # Set positive values to one
        weirDEMDryWetMap062[weirDEMDryWetMap062 > 0] = 1
    
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
        weirDems062[j].append(weirDEMdepthMap062)
        weirDems062_1[j].append(weirDEMDryWetMap062)
        weirDems062_2[j].append(weirDEMdepthMapStacked062)
        dryWetMaps062[j].append(weirDEMDryWetMapStacked062)
    
    del demFile, netCDFFile, zData, zMask062, weirDEMdepthMap062, weirDEMDryWetMap0623, weirDEMDryWetMapStacked062, weirDEMdepthMapStacked062
    del weirDEMDryWetMap062, weirDEMdepthMap0623

#%% Prepare orthos 
# Exp062
orthos062 = [[],[],[],[],[],[]]
weirDems062_stacked = [[],[],[],[],[],[]]
dryWet062_stacked = [[],[],[],[],[],[]]

# Loop through cameras
for j in range(len(orthos062)):
    # Loop through the orthos for Exp062
    for i in range(len(Exp062_orthos)):
        # What is the current ortho folder
        currentOrthoFolder = expExpFolder + '\\Orthos\\' + Exp062_orthos[i]
        # Make list
        orthoList = [file for file in os.listdir(currentOrthoFolder) if not file.endswith('Thumbs.db')]
        orthos = []
        dems = []
        bools = []
        for ortho in orthoList:
            # Load ortho
            currentOrtho = cv2.imread(currentOrthoFolder + '\\' + ortho)
            
            # Apply mask
            # Mask is different for '4636'
            if Exp062_weirs[i] == '4636':
                currentOrtho = currentOrtho*np.flipud(orthoMask3062_4636)
                
            maskedOrtho = np.array(currentOrtho*orthoMask3062[j],dtype='uint8')
                
            # Copy
            maskedOrtho0 = maskedOrtho.copy()
            maskedOrtho1 = maskedOrtho.copy()
            maskedOrtho1[maskedOrtho1>0] = 1
            # Set to floats
            maskedOrtho0 = maskedOrtho0.astype(float)
            maskedOrtho1 = maskedOrtho1.astype(float)

            # Apply Gaussion blur
            if methodO == 'smoothGaus':
                dst0 = cv2.GaussianBlur(maskedOrtho0 ,(smGsettings[0],smGsettings[1]),smGsettings[2],smGsettings[3],borderType = borderTypeO)
                dst1 = cv2.GaussianBlur(maskedOrtho1 ,(smGsettings[0],smGsettings[1]),smGsettings[2],smGsettings[3],borderType = borderTypeO)
            # Divide
            dst = dst0/dst1 
            # Apply mask
            dst[maskedOrtho1==0]=np.nan
            
            # Round values to nearest integer
            dst = np.rint(dst)
            # Continue only if not all values are nan (Because of the additional mask at '4636')
            if not np.all(np.isnan(dst)):
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
                dems.append(weirDems062_2[j][i])
                bools.append(dryWetMaps062[j][i])
                
                del gausOrtho, orthoStacked #, demtemp, boolstemp
            
        # Append stack dem and boolMap
        # If list is not empty
        if orthos:
            orthos062[j].append(orthos)      
            weirDems062_stacked[j].append(dems)
            dryWet062_stacked[j].append(bools)
              
        del ortho, orthos, dems, bools, currentOrtho, maskedOrtho, maskedOrtho0, maskedOrtho1, dst0, dst1, dst,  orthoList

# Stack all data
stacks062 = [[],[],[],[],[],[]]    
stackDems062 = [[],[],[],[],[],[]]  
stackBool062 = [[],[],[],[],[],[]]

for j in range(len(stacks062)):
    # Stack per ortho
    stackO062 = [] # orthos
    stackD062 = [] # dems
    stackB062 = [] # bools
    for i in range(len(orthos062[j])):
        orthoList = orthos062[j][i]
        stackO062.append(np.concatenate(orthoList,axis=0))
        demList = weirDems062_stacked[j][i]
        stackD062.append(np.concatenate(demList,axis=0))
        boolsList = dryWet062_stacked[j][i]
        stackB062.append(np.concatenate(boolsList,axis=0))
    # Concatenate
    stacks062[j] = np.concatenate(stackO062,axis=0)   
    stackDems062[j] = np.concatenate(stackD062,axis=0)   
    stackBool062[j] = np.concatenate(stackB062,axis=0)   

del weirDems062_2, stackO062, stackD062, stackB062, weirDems062_stacked, dryWet062_stacked, dryWetMaps062

print('All data is prepared for ML application and validation')

#%% Loop over concentrations
for c in range(len(concentrations)):
    concentration = concentrations[c]
    max_feature_Clf = max_features_Clf[c]
    max_feature_Regr = max_features_Regr[c]
    print('Computing validations for concentration ' + concentration)
    
    # Determine model directories
    clfFile = modelFolder + "\\Conc" + concentration + "Clf.joblib"
    regrFile = modelFolder + "\\Conc" + concentration + "Regr.joblib"
    valFolder = validationFolder + r"\\Conc" + concentration
   
    # Create folder if it doesn't exist
    if not os.path.exists(valFolder):
        os.makedirs(valFolder)
    
    #% Now load the random forest models
    # classifier model
    print('Loading Clf model for Conc' + concentration)
    dtnow = datetime.datetime.now()
    clf = joblib.load(clfFile)
    dtnow2 = datetime.datetime.now()
    dtnow3 = round((dtnow2 - dtnow).total_seconds())
    print("Clf model loaded for Conc" + concentration + " in " + str(datetime.timedelta(seconds=dtnow3)) + " seconds")

    #regressor model
    print('Loading Regr model for Conc' + concentration)
    dtnow = datetime.datetime.now()
    regr = joblib.load(regrFile)
    dtnow2 = datetime.datetime.now()
    dtnow3 = round((dtnow2 - dtnow).total_seconds())
    print("Regr model loaded for Conc" + concentration + " in " + str(datetime.timedelta(seconds=dtnow3)) + " seconds")
    
    # What information to store?
    recalls062 = []
    precisions062 = []
    F1scores062 = []
    peirceSkillScores062 = []
    rmse062 = []
    bias062 = []
    bias2prmse062 = []
    falseWetPredDepths062 = []
    falseDryActualDepths062 = []
    trueWetPredDepths062 = []
    trueWetActualDepths062 = []

    # Now loop through the cams
    for cam in range(6):
        print("Validating Conc" + concentration + " and cam" + str(cam+1))
        # Append concentration to relevant data
        # Exp062
        recalls062.append('Conc' + concentration + 'Cam' + str(cam))
        precisions062.append('Conc' + concentration + 'Cam' + str(cam))
        F1scores062.append('Conc' + concentration + 'Cam' + str(cam))
        peirceSkillScores062.append('Conc' + concentration + 'Cam' + str(cam))
        rmse062.append('Conc' + concentration + 'Cam' + str(cam))
        bias062.append('Conc' + concentration + 'Cam' + str(cam))
        bias2prmse062.append('Conc' + concentration + 'Cam' + str(cam))
        falseWetPredDepths062.append('Conc' + concentration + 'Cam' + str(cam))
        falseDryActualDepths062.append('Conc' + concentration + 'Cam' + str(cam))
        trueWetPredDepths062.append('Conc' + concentration + 'Cam' + str(cam))
        trueWetActualDepths062.append('Conc' + concentration + 'Cam' + str(cam))

        #%% Apply to Exp062
        # Set the current cam stacked data to validate
        currentStack062 = stacks062[cam]
        currentStackDems062 = stackDems062[cam]
        currentStackBool062 = stackBool062[cam]
        
        # Dry/wet classifier
        clfResult062 = clf.predict(currentStack062)    
        # Regressor
        regrResult062 = regr.predict(currentStack062)
        # Combine Results
        combResult062 = clfResult062*regrResult062
        combResultWet062 = combResult062[clfResult062==1]
        waterDepthsWet062 = currentStackDems062[clfResult062==1]
        
        # We're also interested in the indices of false positives and false negatives
        falseWetClf062 = np.where(np.logical_and(clfResult062 == 1,currentStackBool062 == 0))[0]
        falseDryClf062 = np.where(np.logical_and(clfResult062 == 0,currentStackBool062 == 1))[0]
        # And for the comparison between predicted true depths at actual depths
        trueWetClf062 = np.where(np.logical_and(clfResult062 == 1,currentStackBool062 == 1))[0]
        
        # Calculate relevant scores of the clf model
        recalls062.append(metrics.recall_score(currentStackBool062,clfResult062,average=None))
        precisions062.append(metrics.precision_score(currentStackBool062,clfResult062,average=None))
        F1scores062.append(metrics.f1_score(currentStackBool062,clfResult062,average=None))

        # We're also interested in Peirce skill score, for which we use a different package
        xrclfResultWet062 = xr.DataArray(clfResult062)
        xrbinaryMapWet062 = xr.DataArray(currentStackBool062)
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
        falseDryActualDepths062.append(currentStackDems062[falseDryClf062])
        
        # As well as the true depths (for heatmaps)
        trueWetPredDepths062.append(combResult062[trueWetClf062])
        trueWetActualDepths062.append(currentStackDems062[trueWetClf062])

        print("All data has been validated for Exp062 and Conc" + concentration + " and cam" + str(cam+1))
    
    #%% Write all information to CSV files
    # Exp062
    # Single variable data of training datasets
    with open(valFolder + '\\EV_Conc' + concentration + '_validationData_Exp062.csv', 'w', newline='') as csvFile0620:
        csvFile0620 = csv.writer(csvFile0620,delimiter=",")
        csvFile0620.writerow([['Model_Conc'],['F1Wet'],['F1Dry'],['Peirce'],['RecallWet'],['RecallDry'],
                              ['PrecisionWet'],['PrecisionDry'],['RMSE'],['Bias'],['Bias2FracMSE']])
        for i in range(0,len(F1scores062),2):
            csvFile0620.writerow([[F1scores062[i]],[F1scores062[i+1][0]],[F1scores062[i+1][-1]],[np.atleast_1d(peirceSkillScores062[i+1])[0]],
                                  [recalls062[i+1][0]],[recalls062[i+1][-1]],[precisions062[i+1][0]],[precisions062[i+1][-1]],
                                  [rmse062[i+1]],[bias062[i+1]],[bias2prmse062[i+1]]])
        
    # Predicted water depths from falsely predicted wet cells
    with open(valFolder + '\\EV_Conc' + concentration + '_falseWetPredDepth_Exp062.csv', 'w', newline='') as csvFile0621:
        csvFile0621 = csv.writer(csvFile0621,delimiter=",")
        csvFile0621.writerow([['Model_Conc'],['values']])
        for i in range(0,len(falseWetPredDepths062),2):
            # Trainig datasets
            csvFile0621.writerow([str(falseWetPredDepths062[i])])
            csvFile0621.writerow(falseWetPredDepths062[i+1])
            
    # True water depths from falsely predicted dry cells
    with open(valFolder + '\\EV_Conc' + concentration + '_falseDryTrueDepths_Exp062.csv', 'w', newline='') as csvFile0622:
        csvFile0622 = csv.writer(csvFile0622,delimiter=",")
        csvFile0622.writerow([['Model_Conc'],['values']])
        for i in range(0,len(falseDryActualDepths062),2):
            # Trainig datasets
            csvFile0622.writerow([str(falseDryActualDepths062[i])])
            csvFile0622.writerow(falseDryActualDepths062[i+1])

    # Predicted water depths from correctly predicted wet cells
    with open(valFolder + '\\EV_Conc' + concentration + '_TrueWetPredDepths_Exp062.csv', 'w', newline='') as csvFile0623:
        csvFile0623 = csv.writer(csvFile0623,delimiter=",")
        csvFile0623.writerow([['Model_Conc'],['values']])
        for i in range(0,len(trueWetPredDepths062),2):
            # Trainig datasets
            csvFile0623.writerow([str(trueWetPredDepths062[i])])
            csvFile0623.writerow(trueWetPredDepths062[i+1])
            
    # True water depths from correctly predicted dry cells
    with open(valFolder + '\\EV_Conc' + concentration + '_TrueWetTrueDepths_Exp062.csv', 'w', newline='') as csvFile0624:
        csvFile0624 = csv.writer(csvFile0624,delimiter=",")
        csvFile0624.writerow([['Model_Conc'],['values']])
        for i in range(0,len(trueWetActualDepths062),2):
            # Trainig datasets
            csvFile0624.writerow([str(trueWetActualDepths062[i])])
            csvFile0624.writerow(trueWetActualDepths062[i+1])
        
#%% How long did the run take?
end = datetime.datetime.now()
td = round((end - start).total_seconds())
print("Script running time = " + str(datetime.timedelta(seconds=td)))
