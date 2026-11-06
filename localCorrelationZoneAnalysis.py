'''
This script applies the local correlation zone analysis (Chapters 6-7 in the doctoral dissertation of Eise Nota) from the Water Depth Maps (WDMs) in the
Experimental dataset on estuarine hydrodynamics and morphodynamics for various planforms, perturbations and timescales (https://doi.org/10.5194/essd-2026-427)

Dataset available at: https://doi.org/10.24416/UU01-KKEUY5

This script makes use of the included Python demo to load the WDMs of the different experiments,
with an additional metadata file (on Github) for the experiment-specific information

'''

#%% Clearing all variables
from IPython import get_ipython
get_ipython().magic('reset -sf') 

#%% Set base directory of the dataset network drive (here set to align with the folder structure in the repository)
pwd = r"\\...\original"

import os
os.chdir(pwd) # Setting the functions directory
from heatmapPlots import heatmap, annotate_heatmap

#%% Import packages
import os
import pandas as pd
import datetime
import netCDF4 as nc
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import ast
import itertools

import warnings
warnings.filterwarnings("ignore")

#%% Set base directory of the dataset network drive (here set to align with the folder structure in the repository)

# Set the directory of the functions
start = datetime.datetime.now()

#%% Define fixed variables 
baseModel = '11'        # Base Model within which geometry the data was processed   
gridRes = '25'          # Desired grid resolution in mm; fixed at 25 for water depth maps

#%% Read metadata
metaData = pd.read_excel(pwd + r"\\expMetaData.xlsx") # NOTE this metadata file is on Github

# Which experiments (indices) do we want to compute?
expsi = [0,1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18] # Min 0; Max 20

# Make list of experimental directories
expFolders = os.listdir(pwd + '\\metronome_experiments')

# Loop over the experiments of interest
for expi in expsi:
    print("Computing " + metaData.loc[expi,'Exp'])
    
    #%%Take the experiment-specific metadata
    Expnr = metaData.loc[expi,'Expnr'][:-1]                                                         # Experiment number
    pilot = metaData.loc[expi,'pilot']                                                              # Pilot number, 0 if no pilot
    start_interval = metaData.loc[expi,'start_interval'][:-1]                                       # String of orthomosaic cycle where you want to start # 00000 = first cycle
    end_interval = metaData.loc[expi,'end_interval'][:-1]                                           # String of orthomosaic cycle where you want to end # End with complete numbers as in folder structure 
    conc = metaData.loc[expi,'conc'][1:-1].replace("'", "").split(",")                              # Experimental dye concentration(s)
    cycleDurat = metaData.loc[expi,'cycleDurat']                                                    # Duration of one tidal cycle (s)
    maskExpnr = metaData.loc[expi,'maskExp'][:-1]                                                   # Mask experiment
    maskCycle = metaData.loc[expi,'maskCycle'][:-1]                                                 # Mask cycle
    maskBuffer = metaData.loc[expi,'maskBuffer'][:-1]                                               # Mask buffer
    maskpilot = metaData.loc[expi,'maskpilot']                                                      # Mask pilot
    cyclesToSkip = metaData.loc[expi,'cyclesToSkip'][1:-1].replace("'", "").split(",")              # Cycles not to be included
    cyclesRangesToSkip = metaData.loc[expi,'ignoreCycleRanges'][1:-1].replace("'", "").split(",")   # Already computed, but to be excluded
    varThreshold = metaData.loc[expi,'varThreshold']                                                # Percentage threshold for variance # this value is always 1
    mergeThreshold = metaData.loc[expi,'mergeThreshold']                                            # Overlap threshold for merging zones 
    combcovZones = ast.literal_eval(metaData.loc[expi,'combcovzones'])                              # Final ad hoc assessment: merge zones
    delcovZones = ast.literal_eval(metaData.loc[expi,'delcovzones'])                                # Final ad hoc assessment: delete zones
    
    # Skip iteration if there are no water depth maps
    if int(end_interval) - int(start_interval) == 0:
        continue
    
    # Define label start for relevant folders
    # Experimental folder
    if pilot == 0:
        expStart = 'Exp' + Expnr
    else:
        expStart = 'Exp' + Expnr + '_Pilot' + str(pilot)
    # Mask folder
    if maskpilot == 0:
        maskStart = 'Exp' + maskExpnr
    else:
        maskStart = 'Exp' + maskExpnr + '_Pilot' + str(maskpilot)
    # Define appropriate indices
    expIndex = [expFolders.index(l) for l in expFolders if l.startswith(expStart)][0]
    maskIndex = [expFolders.index(l) for l in expFolders if l.startswith(maskStart)][0]
    
    # Determine mask name
    if maskpilot == 0:
        if int(maskBuffer) > 0:
            maskName = 'MaskExp' + maskExpnr + 'Cycle' + maskCycle + 'Buffer' + maskBuffer
        else:
            maskName = 'MaskExp' + maskExpnr + 'Cycle' + maskCycle
    else:
        if int(maskBuffer) > 0:
            maskName = 'MaskExp' + maskExpnr + 'Pilot' + str(maskpilot) + 'Cycle' + maskCycle + 'Buffer' + maskBuffer
        else:
            maskName = 'MaskExp' + maskExpnr + 'Pilot' + str(maskpilot) + 'Cycle' + maskCycle
    
    #%% Define the relevant folders for mask and Water Depth Maps
    # Current experiment folder
    expFolder = pwd + '\\metronome_experiments\\' + expFolders[expIndex]
    maskFolder = pwd + '\\metronome_experiments\\' + expFolders[maskIndex]
    # Add subsequent folders
    expFolder = expFolder + '\\Overhead Water Depth Maps\\BaseModel' + baseModel + '\\Res' + gridRes + 'mm'
    maskFolder = maskFolder + '\\Masks\\BaseModel' + baseModel + '\\Res' + gridRes + 'mm'
    # writeFolder
    writeFolder = pwd + '\\writeFolder\\'
    
    # Create a directory if it does not exist
    if not os.path.exists(writeFolder):
        os.makedirs(writeFolder)  
    
    # Defining the depthMapFolders based on the amount of available dye concentrations
    depthMapFolders = []
    for c in conc:
        depthMapFolders.append(expFolder + '\\Conc' + c + '\\' + maskName.replace('Exp',''))
    
    #%% Load mask
    maskDir = maskFolder + '\\' + maskName + '.nc'
    maskNC = nc.Dataset(maskDir, 'r')
    # Extract the relevant information
    mask = np.array(maskNC.get_variables_by_attributes(name='mask')[0])
    xAxis = np.array(maskNC.get_variables_by_attributes(name='X-axis')[0])
    yAxis = np.array(maskNC.get_variables_by_attributes(name='Y-axis')[0])
    # Convert the X/Y-axis information to plottable values
    xValues = np.arange(xAxis[0],xAxis[1],xAxis[2])
    yValues = np.arange(yAxis[0],yAxis[1],yAxis[2])
    xx, yy = np.meshgrid(xValues,yValues)
    # Flip yy
    yy = np.flipud(yy)
    # We can close the file again
    maskNC.close()
    # Flip vertically for ortho
    mask = np.flipud(mask)
    # Make mask suited for 3D
    orthoMask0 = np.zeros([np.size(mask,0),np.size(mask,1),3])
    orthoMask0[:,:,0] = mask; orthoMask0[:,:,1] = mask; orthoMask0[:,:,2] = mask
    
    # Set zeros to nan
    nanmask = mask.astype(float)
    nanmask[nanmask==0] = np.nan
    # Create mesh with deleted columns with all nans
    xxnan = xx[:,~np.all(np.isnan(nanmask), axis=0)]
    yynan = yy[:,~np.all(np.isnan(nanmask), axis=0)]
    # Where do the cuts occur?
    xLeft = np.where(xx[0,:]==xxnan[0,0])[0][0]
    xRight = np.where(xx[0,:]==xxnan[0,-1])[0][0]+1
    
    # Define the cut as we only want to study downstream half
    # Where is value closest to 10?
    xCut = (np.abs(xx[0,:] - 10)).argmin()
    xCut2 = (np.abs(xx[0,:] - 10)).argmin()
    # Also update xAxis and xValues
    xValues = xValues[xCut2:]    
    xAxis[0] = xValues[0]
    # Update mask
    nanmask[:,:xCut] = 0
    # Set zeros again to nan
    nanmask[nanmask==0] = np.nan
    xxCut = xx[:,xCut:xRight]
    yyCut = yy[:,xCut:xRight]
    nanmaskCut = nanmask[:,xCut:xRight]
    xCutAxis = np.array([xxCut[0,0],xxCut[-1,-1]+xAxis[2]/2,xAxis[2]])
    yCutAxis = np.array([yyCut[-1,-1],yyCut[0,0]+yAxis[2]/2,yAxis[2]])
    
    #%% Define the relevant Depth Maps and their information
    # Update relevant dye concentrations for Exp063 (cycles ranges of 1Hz imagery and varying dye concentrations are excluded from the interval)
    if Expnr == '063' and pilot == 0:
        depthMapFolders = depthMapFolders[:1]
    
    # Determine the available files in the folder
    avail_files = []
    avail_fileDirs = []
    # Create list of available depth maps
    for depthMapFolder in depthMapFolders:
        avail_files.append(os.listdir(depthMapFolder))
        avail_fileDirs.append([os.path.join(depthMapFolder, file) for file in os.listdir(depthMapFolder)])
    # Stack avail_files to single list
    avail_files = sum(avail_files, [])   
    avail_fileDirs = sum(avail_fileDirs, [])   
    
    # Determine relevant depth maps
    rel_depthMaps = []
    rel_timesteps = []
    for i in range(len(avail_files)):
        file = avail_files[i]
        # First test whether the cycles occur in the cyclesToSkip list
        # Cycle number occurs at [-8:-3]
        if file[-8:-3] not in cyclesToSkip:
            if int(start_interval) <= int(file[-8:-3]) <= int(end_interval): 
                rel_depthMaps.append(avail_fileDirs[i])
                rel_timesteps.append(int(file[-8:-3]))

    rel_timesteps = np.array(rel_timesteps)
    rel_timestamps = rel_timesteps * cycleDurat # seconds
    # Sort
    rel_depthMaps = sorted(rel_depthMaps, key=lambda x: x.split('_')[2])
    rel_timesteps = sorted(rel_timesteps)
    rel_timestamps = sorted(rel_timestamps)

    # Do a final filtering on cyclesRangesToSkip, in case they are set
    if len(cyclesRangesToSkip[0]) != 0:
        # Reversed for loop to maintain proper indices, so "-1,-1,-1)"
        for i in range(len(rel_timesteps)-1,-1,-1):
            for currentRange in cyclesRangesToSkip:
                if int(currentRange[:5]) <= rel_timesteps[i] < int(currentRange[-5:]):
                    rel_timesteps.pop(i)
                    rel_depthMaps.pop(i)
                    rel_timestamps.pop(i)
        
    #%% Load all depth Maps into a single list
    depthMaps = []
    # Loop over the relevant depthMaps
    for i in range(len(rel_depthMaps)):
        currentDepthMap = rel_depthMaps[i]
        # Load currentDepthMap
        netCDFFile = nc.Dataset(currentDepthMap)
        # Extract the relevant information
        depthData = np.array(netCDFFile.get_variables_by_attributes(name='Z-axis')[0])
        # Reshape to align with plottable data
        depthData = depthData[:,:,0].reshape((depthData.shape[0],depthData.shape[1]))
        # We can close the file again
        netCDFFile.close()
        # Append
        depthMaps.append(depthData)
        # Project depthData to the full coordinate system
        depthData = depthData[:,xCut-xLeft:xRight]
        
    depthMaps = np.array(depthMaps)*nanmaskCut   
    
    #%% Now conduct the desired analysis on the Water Depth Maps
    # Restructure depthMaps to shape (n_samples, n_features), i.e. (n_timesteps, cells)
    depthMaps2d = depthMaps.reshape(len(depthMaps),-1) # Note this iterates over rows
    # Make a copy
    copyMap = depthMaps2d.copy()
    # Remove columns with all NaNs
    depthMaps2d = depthMaps2d[:,~np.all(np.isnan(depthMaps2d),axis=0)]
    # dMin = np.min(depthMaps2d,axis=0)
    # dMax = np.max(depthMaps2d,axis=0)
    # depthMaps2d = (depthMaps2d - dMin)/(dMax-dMin)
    # # If a full timeseries is dry, then the above results in dry
    # # Set those timesteps back to zeros
    # depthMaps2d[:,np.all(np.isnan(depthMaps2d),axis=0)] = 0     
        
    #%% Compute/load the variance matrix in a single for loop
    # Population size
    popsize = len(depthMaps2d.T)
    
    varFile = expStart + '_varMatrix.nc'
    # Compute variance matrix if it doesn't exist
    if not os.path.exists(writeFolder + varFile):
        start0 = datetime.datetime.now()
        print('Compute ' + varFile)
        varValues = np.zeros((popsize,popsize))*np.nan
        # Loop
        for i in range(popsize):
            if not i%500:
                print(str((i/popsize)*100)[:4] + '%')
            # Broadcast info
            currentDifference = depthMaps2d.T[i:, None] - depthMaps2d.T[i,:] 
            currentVar = np.var(currentDifference,axis=2)
            # Only fill on one side, no duplicates
            #varValues[i,i:] = currentVar.T
            varValues[i:,i] = currentVar.T
            del(currentDifference)
        end0 = datetime.datetime.now()
        tddtw = round((end0 - start0).total_seconds())
        print(expStart + " variance matrix computed in " + str(timedelta(seconds=tddtw)) + " seconds")    
        
        # Write to NetCDF
        varFile = expStart + '_varMatrix.nc'
        netCDFFile = nc.Dataset(writeFolder + varFile,'w') 
        netCDFFile.title = varFile[:-3]
        # Add axes attributes
        netCDFFile.createDimension('varrange', len(varValues))
        # Add information
        var_var = netCDFFile.createVariable('var', np.float64, ('varrange','varrange'))
        var_var[:] = varValues
        # Now we can close the netCDF file
        netCDFFile.close()
        print(varFile + ' is stored on the RDL')
    else: # load the file
        start0 = datetime.datetime.now()
        print('Load ' + varFile)
        netCDFFile = nc.Dataset(writeFolder + varFile,'r') 
        varValues = np.array(netCDFFile.get_variables_by_attributes(name='var')[0])
        # Now we can close the netCDF file
        netCDFFile.close()
        end0 = datetime.datetime.now()
        td0 = round((end0 - start0).total_seconds())
        print("variance matrix loaded in " + str(timedelta(seconds=td0)) + " seconds")
    
    #%% Set diagonal to nan, we don't want zeros here of highest correlation
    np.fill_diagonal(varValues,np.nan)
    
    # Define indices for var
    varindices = np.arange(0,len(varValues))
    
    # What are the minimum variance values for each of the cells?
    minVars = np.nanmin(varValues,axis=0)
    # Get the order of sorted maximum correlations
    minsortsArgs = np.argsort(minVars)
    
    # Current writeFile
    fileName = 'var' + "{:.2f}".format(varThreshold).replace(".","_") + '_merge' + str(mergeThreshold)
    writeFile = fileName + '.nc'
    # Only compute if writeFile doesn't already exist
    if not os.path.exists(writeFolder + writeFile):
        print('Computing variance threshold: ' + str(varThreshold) + '%; and merge threhsold: ' + str(mergeThreshold) + '%')
        
        # Assign the varThreshold for clusters. ## maybe with a nonzero max (maxsorts[maxsorts>0]), maybe not
        clusterThreshold = np.nanpercentile(varValues,varThreshold)
        
        # Now start assiginging clusters by number
        clusterzones = []
        clusterzonesA = np.array(clusterzones)
        
        print('Assigning zones')
        # Loop over the minsortsArgs
        for j in range(len(minsortsArgs)):
            if not j%1000:
                print(str((j/popsize)*100)[:4] + '%')
            currentIndex = minsortsArgs[j]
            # Test whether current cell is already in a cluster
            if ~np.isin(currentIndex, clusterzonesA):
                currentCell = varValues[:,minsortsArgs[j]]
                zoneCells = currentCell < clusterThreshold
                if np.sum(zoneCells) > 0:
                    # Which covariances exceed the threshold
                    cluster = varindices[currentCell < clusterThreshold]
                    # Insert current cell index as well
                    cluster = np.sort(np.append(cluster,currentIndex))
                    # Do cells of the new cluster already occur in the clusterzone?
                    newCluster = 1
                    for czi in range(len(clusterzones)):
                        cz = clusterzones[czi]
                        # Let's say that if more than half of these cells already occur in a cluster, then we're going to include
                        if len(np.intersect1d(cluster, cz)) > int(np.ceil(len(cluster)*(mergeThreshold/100))) or len(np.intersect1d(cluster, cz)) > int(np.ceil(len(cz)*(mergeThreshold/100))):
                            newCluster = 0
                            # Combine into existing cluster
                            clusterzones[czi] = np.unique(np.append(cluster,cz))
                            break
                    if 'cz' in locals():
                        del(cz)
                    # Append new cluster
                    if newCluster == 1:
                        clusterzones.append(cluster)
                    # Make to 1d for testing array    
                    clusterzonesA = np.unique(np.concatenate(clusterzones).ravel())
                    del(cluster)
                
                del(currentCell, zoneCells)
        print('100%')
        del(clusterzonesA)
        
        #%% Do another iteration through the cluster to determine residual overlap overreaching the threshold
        print('Determining residual overlap')
        whilei = 0
        while (len(np.unique(np.concatenate(clusterzones)))/len(np.concatenate(clusterzones)))*100 < 100-mergeThreshold:
            clusterzones0 = []
            clusterzonesB = []
            for j in range(len(clusterzones)):
                if ~np.isin(j, clusterzonesB):
                    cluster = clusterzones[j]
                    for czi in range(j+1,len(clusterzones)):
                        if ~np.isin(czi, clusterzonesB):
                            cz = clusterzones[czi]
                            if len(np.intersect1d(cluster, cz)) > int(np.ceil(len(cluster)*(mergeThreshold/100))) or len(np.intersect1d(cluster, cz)) > int(np.ceil(len(cz)*(mergeThreshold/100))):
                                # Combine into existing cluster
                                cluster = np.unique(np.append(cluster,cz))
                                # Store index in clusterzonesB
                                clusterzonesB.append(czi)        
                    # Append new zone
                    clusterzones0.append(cluster)
            # Update clusterzones
            clusterzones = clusterzones0.copy()
            whilei += 1
        print('While loop ran for ' + str(whilei) + ' iterations')
        if 'clusterzones0' in locals():
            del(clusterzones0, clusterzonesB, cluster, cz)

        # Assign correlation clusters # Let 0 be clusterless cells
        print('Nr. of correlation clusters: ' + str(len(clusterzones)))
        # Continue only if there are a max of 12 clusterzones
        # If there are more clusters, keep the largest 12
        if len(clusterzones)>12:
            print('Maintaining 12 largest clusters')
            clusterzones = sorted(clusterzones, key=len, reverse=True)
            clusterzones = clusterzones[:12]

        corclusters = np.arange(1,len(clusterzones)+1)
        
        #%% If set, delete clusters
        if len(delcovZones) > 0:
            print('Delete set clusters')
            # loop through delcovZones
            for zone in delcovZones:
                # First set to nan to maintain indices
                clusterzones[np.where(corclusters==zone)[0][0]] = np.nan
                # Set original cluster to 0
                corclusters[np.where(corclusters==zone)[0][0]] = 0
                
        #%% If set, merge clusters
        if len(combcovZones) > 0:
            print('Merge set clusters')
            mergeClusters = []
            # loop through delcovZones
            for combzones in combcovZones:
                mergeClusters.append([])
                # loop through combzones
                #if len(np.where(corclusters==zone)[0]) > 0:
                for zone in combzones:
                    mergeClusters[-1].append(clusterzones[np.where(corclusters==zone)[0][0]])
                    # Set original cluster to 0
                    corclusters[np.where(corclusters==zone)[0][0]] = 0
            # Loop trhough mergeclusters to flatten
            for ii in range(len(mergeClusters)):
                mergeClusters[ii] = np.concatenate(mergeClusters[ii])
        
        #%% Create new clusterzones
        clusterzonesNew = []
        # Loop through original clusters
        for ii in range(len(corclusters)):
           # Original cluster that is not merged
           if corclusters[ii] > 0:
               clusterzonesNew.append(clusterzones[ii])
        # Now append merged clusters
        if len(combcovZones) > 0:
            for cluster in mergeClusters:
                clusterzonesNew.append(cluster)
        # Sort clusterzonesNew 
        clusterzonesNew = sorted(clusterzonesNew, key=len, reverse=True)
        # Create new corclusters
        corclustersNew = np.arange(1,len(clusterzonesNew)+1)
        
        # Overwrite
        clusterzones = clusterzonesNew.copy()
        corclusters = corclustersNew.copy()
        
        del(clusterzonesNew, corclustersNew)
        
        #%% Make zoneMap
        zoneMaps = []
        zoneMap = depthMaps2d[0,:]*0
        zoneMapC = (depthMaps2d[0,:]*0).copy()
        for cl in corclusters:
            # Aggregated zone map
            zoneMap[clusterzones[cl-1]] = cl
            # ZoneMaps list
            zoneMapC[clusterzones[cl-1]] = cl
            # Put back in the copyMap mold
            copyMapC = copyMap[0,:].copy()*0
            copyMapC[~np.isnan(copyMapC)] = zoneMapC
            # Project back to 2d
            zoneMapNew = copyMapC.reshape(np.shape(depthMaps[0,:,:]))
            zoneMaps.append(zoneMapNew)
            zoneMapC = (depthMaps2d[0,:]*0).copy()
        # Put back in the copyMap mold
        copyMapC = copyMap[0,:].copy()*0
        copyMapC[~np.isnan(copyMapC)] = zoneMap
        # Project back to 2d
        zoneMapNew = copyMapC.reshape(np.shape(depthMaps[0,:,:]))
        plt.imshow(zoneMapNew)
        del(copyMapC, zoneMap, zoneMapC)
        
        #%% How many overlapping regions may there be?
        overlapRegions = []
        # How many possible overlapping regions are there?
        for comb in range(len(corclusters),1,-1): # At least paris of 2
            overlapRegions.append(list(itertools.combinations(corclusters, comb)))
        # Determine the intersection duos
        # if there are any
        if len(overlapRegions) > 0:
            # The first overlapRegion includes all region
            for comb in overlapRegions[0]:
                # Which zones to study?
                zones2Study = []
                for c in comb:
                    zones2Study.append(clusterzones[c-1]) # -1 for proper index
                # Find the overlap between 
                intersects = []
                intersectzones = []
                for z in range(len(zones2Study)-1):
                    #print(z)
                    for z2 in range(z+1,len(zones2Study)):
                        #print(z2)
                        intersects.append(np.intersect1d(zones2Study[z],zones2Study[z2]))
                        intersectzones.append((z+1,z2+1))
            # Determine the overlapping cells
            # concatenate info into single list
            overlapRegions = list(itertools.chain.from_iterable(overlapRegions))
            overlapCells = []
            # Loop through the regions
            for l in range(len(overlapRegions)):
                currentRegion = overlapRegions[l]
                potentialRegionZones = []
                # Loop through intersect zones whether both intersect zones occur here
                for iz in range(len(intersectzones)):
                    ciz = intersectzones[iz]
                    if set(ciz).issubset(currentRegion):
                        potentialRegionZones.append(intersects[iz])
                # Now we only need the cells that occur in all potentialRegionZones
                result = set(potentialRegionZones[0])
                # Find intersection with remaining sub-arrays
                for sub_array in potentialRegionZones[1:]:
                    result.intersection_update(sub_array)
                result = np.array(list(result))
                # For further overlap regions, remove pre-existing cells
                if l > 0:
                    for oc in overlapCells:
                        result = np.array(list(set(result) - set(oc)))
                # Remove zone if empty
                if len(result) == 0:
                    overlapRegions[l] = []
                overlapCells.append(result)
                del(result, potentialRegionZones)
            del(intersects, intersectzones)
            
            # Adjust the information into somtehing that can be stored into netcdf
            oRegions = []
            oCells = []
            # If there is only one region, then ad a dimension to the overlaps
            if len(overlapRegions) == 0:
                overlapRegions.append([])
                overlapCells.append([])
            # Store only the non-empty zones
            for j in range(len(overlapCells)):
                if len(overlapCells[j]) > 0: 
                    oRegions.append(overlapRegions[j])
                    oCells.append(overlapCells[j])
            del(overlapRegions, overlapCells)
            # If there is no overlap, set to nan
            if len(oCells) == 0:
                oRegionsArray = [[np.nan]]
                oCellsArray = [[np.nan]]
            elif len(oCells[0]) == 0:
                oRegionsArray = [[np.nan]]
                oCellsArray = [[np.nan]]
            # In case of multiple overlapping regions, a homogenous 2d array needs to be created
            else:
                # What is the largest item?
                maxRegions = np.max([len(sub) if type(sub) != int else 1 for sub in oRegions])
                maxCells = np.max([len(sub) if type(sub) != int else 1 for sub in oCells])
                # Create empty 2d arrays to store the information
                oRegionsArray = np.zeros((len(oRegions),maxRegions))*np.nan
                oCellsArray = np.zeros((len(oCells),maxCells))*np.nan
                # Fill values
                for j in range(len(oRegions)):
                    # current lengths
                    orLength = len(oRegions[j])
                    ocLength = len(oCells[j])
                    # Fill
                    oRegionsArray[j,:orLength] = oRegions[j]
                    oCellsArray[j,:ocLength] = oCells[j]  
            # Final flip
            oRegionsArray = np.flipud(oRegionsArray)
            oCellsArray = np.flipud(oCellsArray)
        else:
            oRegionsArray = [[np.nan]]
            oCellsArray = [[np.nan]]
        
        #%% Now we want to extract some information for each of these high variance zones
        # Some timeseries statistics
        print('Compute zone statstics')
        meanDepths = []
        medianDepths = []
        maxDepths = []
        minDepths = []
        stdDepths = []
        clusterComparisonInfo = []
        clusterComparison = [] # e.g. if variances are strongly negative, then they'll likely move counter phase
        for ci in range(len(corclusters)):
            if len(corclusters) > 1:
                print(str((ci/(len(corclusters)-1))*100)[:4] + '%')
            else:
                print('100%')
            # Make a zone binary map yielding 1 for each zone cell
            zoneBinary = zoneMaps[ci] == corclusters[ci]
            # Take the timeseries
            zoneSeries = depthMaps[:,zoneBinary]
            # Store statistics
            meanDepths.append(np.mean(zoneSeries,axis=1))
            medianDepths.append(np.median(zoneSeries,axis=1))
            stdDepths.append(np.std(zoneSeries,axis=1))
            # np.min and np.max don't work if all values are nan
            if np.isnan(meanDepths[-1]).all():
                maxDepths.append(meanDepths[-1])
                minDepths.append(meanDepths[-1])
            else:
                maxDepths.append(np.max(zoneSeries,axis=1))
                minDepths.append(np.min(zoneSeries,axis=1))
            # Compare clusters, also with themselves
            for c2i in range(0,ci+1):
                clusterComparisonInfo.append([corclusters[c2i],corclusters[ci]])
                # We're going to calculate the pearson correlation coefficient between the zone duos
                zoneBinary2 = zoneMapNew == corclusters[c2i]
                # Take the timeseries
                zoneSeries2 = depthMaps[:,zoneBinary2]
                if len(zoneSeries[0]) == 0 or len(zoneSeries2[0]) == 0:
                    meanPears = np.nan
                    medianPears = np.nan
                    stdPears = np.nan
                    maxPears = np.nan
                    minPears = np.nan
                    clusterComparison.append([meanPears,medianPears,maxPears,minPears,stdPears])
                else:
                    # Calculate Pearson correlation coefficient
                    pears = np.corrcoef(zoneSeries.T,zoneSeries2.T)
                    # The two variables in pears are aggregated, 
                    # so we're only interested in the statistics of one fourth of the pears
                    # this is either lower left corner or upper right corner
                    pears = pears[len(zoneSeries.T):,:len(zoneSeries.T)]
                    
                    # Store statistics
                    meanPears = np.nanmean(pears)
                    medianPears = np.nanmedian(pears)
                    stdPears = np.nanstd(pears)
                    # np.min and np.max don't work if all values are nan
                    if np.isnan(meanPears):
                        maxPears = np.nan
                        minPears = np.nan
                    else:
                        maxPears = np.nanmax(pears)
                        minPears = np.nanmin(pears)
                    
                    clusterComparison.append([meanPears,medianPears,maxPears,minPears,stdPears])
                
                    del(pears, zoneSeries2, zoneBinary2)
            
            del(zoneSeries)
            
        # Transform to arrays
        meanDepths = np.array(meanDepths)
        medianDepths = np.array(medianDepths)
        maxDepths = np.array(maxDepths)
        minDepths = np.array(minDepths)
        stdDepths = np.array(stdDepths)
        clusterComparisonInfo = np.array(clusterComparisonInfo)
        clusterComparison = np.array(clusterComparison)
        
        #%% Write to NetCDF
        netCDFFile = nc.Dataset(writeFolder + writeFile,'w') 
        netCDFFile.title = writeFile[:-3]
        # Add axes attributes
        netCDFFile.createDimension('thresholdrange', len([clusterThreshold]))
        netCDFFile.createDimension('yrange', len(zoneMapNew[:,0]))
        netCDFFile.createDimension('xrange', len(zoneMapNew[0,:]))
        netCDFFile.createDimension('coordsrange', len(xCutAxis))
        netCDFFile.createDimension('timerange', len(rel_timesteps))
        netCDFFile.createDimension('clusterrange', len(corclusters))
        netCDFFile.createDimension('clustercomprange', len(clusterComparisonInfo))
        netCDFFile.createDimension('clustercomprange2', len(clusterComparisonInfo[0]))
        netCDFFile.createDimension('clustercompstudyrange', len(clusterComparison[0]))
        netCDFFile.createDimension('overlaprange', len(oRegionsArray))
        netCDFFile.createDimension('overlapregionsrange', len(oRegionsArray[0]))
        netCDFFile.createDimension('overlapcellsrange', len(oCellsArray[0]))
        # Add information
        threshold_var = netCDFFile.createVariable('threshold', np.float64, ('thresholdrange'))
        threshold_var[:] = [clusterThreshold]
        zoneMap_var = netCDFFile.createVariable('ZoneMap', np.float64, ('yrange','xrange'))
        zoneMap_var[:] = zoneMapNew
        xAxis_var = netCDFFile.createVariable('X-axis', np.float64, ('coordsrange'))
        xAxis_var[:] = xCutAxis
        yAxis_var = netCDFFile.createVariable('Y-axis', np.float64, ('coordsrange'))
        yAxis_var[:] = yCutAxis
        time_var = netCDFFile.createVariable('timesteps', np.int32, ('timerange'))
        time_var[:] = np.array(rel_timesteps)
        cluster_var = netCDFFile.createVariable('clusters', np.int32, ('clusterrange'))
        cluster_var[:] = corclusters
        meandepth_var = netCDFFile.createVariable('meanClusterDepth', np.float64, ('clusterrange','timerange'))
        meandepth_var[:] = meanDepths
        mediandepth_var = netCDFFile.createVariable('medianClusterDepth', np.float64, ('clusterrange','timerange'))
        mediandepth_var[:] = medianDepths
        maxdepth_var = netCDFFile.createVariable('maxClusterDepth', np.float64, ('clusterrange','timerange'))
        maxdepth_var[:] = maxDepths
        mindepth_var = netCDFFile.createVariable('minClusterDepth', np.float64, ('clusterrange','timerange'))
        mindepth_var[:] = minDepths
        stddepth_var = netCDFFile.createVariable('stdClusterDepth', np.float64, ('clusterrange','timerange'))
        stddepth_var[:] = stdDepths
        clustercompInfo_var = netCDFFile.createVariable('clusterCompInfo', np.int32, ('clustercomprange','clustercomprange2'))
        clustercompInfo_var[:] = clusterComparisonInfo
        clustercomp_var = netCDFFile.createVariable('clusterComp', np.float64, ('clustercomprange','clustercompstudyrange'))
        clustercomp_var[:] = clusterComparison
        overlapregion_var = netCDFFile.createVariable('overlapRegions', np.float64, ('overlaprange','overlapregionsrange'))
        overlapregion_var[:] = oRegionsArray
        overlapcells_var = netCDFFile.createVariable('overlapCells', np.float64, ('overlaprange','overlapcellsrange'))
        overlapcells_var[:] = oCellsArray
        # Now we can close the netCDF file
        netCDFFile.close()
        print(writeFile + ' is stored on the RDL')
        
        del(meanDepths, medianDepths, maxDepths, minDepths, stdDepths, zoneMapNew)
            
    del(copyMap, rel_timesteps, minVars, varValues, minsortsArgs)
    
    #%% Load and plot                
    print('Load ' + writeFile)
    netCDFFile = nc.Dataset(writeFolder + writeFile,'r') 
    threshold = np.array(netCDFFile.get_variables_by_attributes(name='threshold')[0])[0]
    rel_cycles = np.array(netCDFFile.get_variables_by_attributes(name='timesteps')[0])
    zoneMap = np.array(netCDFFile.get_variables_by_attributes(name='ZoneMap')[0])
    xAxis = np.array(netCDFFile.get_variables_by_attributes(name='X-axis')[0])
    yAxis = np.array(netCDFFile.get_variables_by_attributes(name='Y-axis')[0])
    corclusters = np.array(netCDFFile.get_variables_by_attributes(name='clusters')[0])
    meanDepths = np.array(netCDFFile.get_variables_by_attributes(name='meanClusterDepth')[0])
    medianDepths = np.array(netCDFFile.get_variables_by_attributes(name='medianClusterDepth')[0])
    maxDepths = np.array(netCDFFile.get_variables_by_attributes(name='maxClusterDepth')[0])
    minDepths = np.array(netCDFFile.get_variables_by_attributes(name='minClusterDepth')[0])
    stdDepths = np.array(netCDFFile.get_variables_by_attributes(name='stdClusterDepth')[0])
    clusterComparisonInfo = np.array(netCDFFile.get_variables_by_attributes(name='clusterCompInfo')[0])
    clusterComparison = np.array(netCDFFile.get_variables_by_attributes(name='clusterComp')[0])
    overlapRegions = np.array(netCDFFile.get_variables_by_attributes(name='overlapRegions')[0])
    overlapCells = np.array(netCDFFile.get_variables_by_attributes(name='overlapCells')[0])
    
    # Now we can close the netCDF file
    netCDFFile.close()
    
    xValues = np.arange(xAxis[0],xAxis[1],xAxis[2])
    yValues = np.arange(yAxis[0],yAxis[1],yAxis[2])
    xx, yy = np.meshgrid(xValues,yValues)
    
    # region ticks to plot
    plotTicks = []
    for cluster in corclusters:
        plotTicks.append(str(cluster))
    
    #%% Prepare the overlap regions
    if len(overlapRegions[0]) >=2:
        # Make 2d and 1d copymap
        copyMap1d = zoneMap.copy().ravel()*0
        for oc in range(len(overlapCells)):
            # delete nans and set to integers
            currentCells = overlapCells[oc][~np.isnan(overlapCells[oc])].astype(int)
            # Get true cells
            trueCells = copyMap1d[~np.isnan(copyMap1d)]
            # Fill with overlapping regions
            trueCells[currentCells] = np.max(corclusters)+oc+1
            # Insert in copyMap
            copyMap1d[~np.isnan(copyMap1d)] = trueCells
            # Remove nans in current regions
            currentRegions = overlapRegions[oc][~np.isnan(overlapRegions[oc])]
            # Append to plotTicks
            currentRegion = str(currentRegions.astype(int).astype(str))
            currentRegion = currentRegion.replace(r"['","").replace(r"']","").replace(r"' '","-")
            plotTicks.append(currentRegion)
            del(currentCells, trueCells)
        # Put back in the 2d mold
        overlapMap = copyMap1d.reshape(np.shape(zoneMap))
        # Include in zoneMap
        zoneMap[overlapMap>0] = overlapMap[overlapMap>0]
    
    #%% Plot statements
    figWidth = 20
    figHeight = figWidth*(3.5/(xx[-1,-1]-xx[0,0]))
    figsize=(figWidth, figHeight) # scale y- and x-axes
    labelsize = 30 # for labels x-, y-axes and colorbar
    
    # Colorbar information
    cbzones = np.arange(1,int(np.nanmax(zoneMap)+1))
    if np.nanmin(zoneMap) == 0:  
        # get discrete colormap
        cmap = plt.get_cmap('viridis', np.max(cbzones) + 1)
    else: # Each cell has a zone
        # get discrete colormap
        cmap = plt.get_cmap('viridis', np.max(cbzones))
    # Update cbticks
    bczonesize = cbzones[-1]/(cbzones[-1]+1)
    cbticks = np.arange(bczonesize+bczonesize/2,bczonesize*(cbzones[-1]+1),bczonesize)
    
    #%% Plot the zone Map
    output_file = writeFolder + '\\' + fileName + '.png'
   
    fig, ax = plt.subplots(figsize = figsize,layout='constrained') # Mannually determing larger figsize. Optionally also incorporate this as input variable
    im = ax.pcolormesh(xx, yy, np.flipud(zoneMap),
                    cmap=cmap,
                    rasterized=True)
    ax.axis('equal')
    ax.set_yticks([0,1,2,3])
    ax.set_xticks([10,12,14,16,18])
    ax.set_ylabel('y (m)', size=35)
    ax.set_xlabel('x (m)', size=35)
    ax.set_yticklabels(['0','1','2','3'],fontsize=25)
    ax.set_xticklabels(['10','12','14','16','18'],fontsize=25)
    ax.set_title(fileName + '% (' + str(threshold) + ')' ,fontsize=40)
    # Add colorbar
    clb = fig.colorbar(im, ax=ax,location='right', pad = 0.005, shrink = 0.9, ticks=cbticks)
    clb.set_label('Correlation zones', size=35)
    clb.ax.tick_params(labelsize=30)
    clb.ax.set_yticklabels(plotTicks)  # vertically oriented colorbar
    
    # Store
    plt.savefig(output_file, bbox_inches='tight', dpi=fig.dpi)
    plt.show()
    
    #%% plot the zone statistics 
    writeFolderC = writeFolder + fileName.split("_")[-3] + '_' + fileName.split("_")[-2] + '_' + fileName.split("_")[-1]
    # Create a directory if it does not exist
    if not os.path.exists(writeFolderC):
        os.makedirs(writeFolderC)
    
    # Plot statements
    figWidth = 20
    figHeight = figWidth
    figsize=(figWidth, figHeight) # scale y- and x-axes
    labelsize = 30 # for labels x-, y-axes and colorbar
    
    # Loop through zones
    for c in range(len(corclusters)):
        fileImage = fileName + '_depthStatistics_zone' + str(corclusters[c]) + '.png'
        fileTitle = fileImage[:-4].replace("_"," ")
        output_file = writeFolderC + '\\' + fileImage
       
        fig, ax = plt.subplots(5,1, figsize = figsize,layout='constrained',sharex=True) # Mannually determing larger figsize. Optionally also incorporate this as input variable
        # mean Depth
        ax[0].plot(rel_cycles,meanDepths[c,:]*100)
        ax[0].set_ylabel('mean h (cm)', size=35)
        ax[0].tick_params(axis='y', labelsize=25)
        # median Depth
        ax[1].plot(rel_cycles,medianDepths[c,:]*100)
        ax[1].set_ylabel('median h (cm)', size=35)
        ax[1].tick_params(axis='y', labelsize=25)
        # max Depth
        ax[2].plot(rel_cycles,maxDepths[c,:]*100)
        ax[2].set_ylabel('max h (cm)', size=35)
        ax[2].tick_params(axis='y', labelsize=25)
        # min Depth
        ax[3].plot(rel_cycles,minDepths[c,:]*100)
        ax[3].set_ylabel('min h (cm)', size=35)
        ax[3].tick_params(axis='y', labelsize=25)
        # std Depth
        ax[4].plot(rel_cycles,stdDepths[c,:]*100)
        ax[4].set_ylabel('std h (cm)', size=35)
        ax[4].set_xlabel('Tidal cycles', size=35)
        ax[4].tick_params(axis='both', labelsize=25)
        # Title
        fig.suptitle(fileTitle,fontsize=50)

        # Store
        plt.savefig(output_file, bbox_inches='tight', dpi=fig.dpi)
        plt.show()
  
    #%% Plot comparison of zones
    # Plot statements
    figWidth = 30
    figHeight = figWidth/2
    figsize=(figWidth, figHeight) # scale y- and x-axes
    labelsize = figWidth*3/len(corclusters) # for labels x-, y-axes and colorbar
    textsize = figWidth*2.5/len(corclusters)
    titlesize = figWidth*3.5/len(corclusters)
    
    fileImage = fileName + '_zones_pearson.png'
    output_file = writeFolderC + '\\' + fileImage
    fileTitle = fileImage[:-4].replace("_"," ")
    
    # We'll make heatmaps with values in it
    meanMap = np.zeros((len(corclusters),len(corclusters)))
    medianMap = np.zeros((len(corclusters),len(corclusters)))
    maxMap = np.zeros((len(corclusters),len(corclusters)))
    minMap = np.zeros((len(corclusters),len(corclusters)))
    
    for r in range(len(corclusters)):
        # Only need to loop over new columns
        for c in range(r,len(corclusters)):
            # Which row and column are we in?
            clusterR = corclusters[r]
            clusterC = corclusters[c]
            # Where in clusterComparisonInfo is this?
            for ii in range(len(clusterComparisonInfo)):
                if clusterR == clusterComparisonInfo[ii,:][0] and clusterC == clusterComparisonInfo[ii,:][1]:
                    # Fill information
                    meanMap[r,c] = clusterComparison[ii,:][0]
                    medianMap[r,c] = clusterComparison[ii,:][1]
                    maxMap[r,c] = clusterComparison[ii,:][2]
                    minMap[r,c] = clusterComparison[ii,:][3]
                    # Also fill on the opposite side if row doesn't equal column
                    if r != c:
                        meanMap[c,r] = clusterComparison[ii,:][0]
                        medianMap[c,r] = clusterComparison[ii,:][1]
                        maxMap[c,r] = clusterComparison[ii,:][2]
                        minMap[c,r] = clusterComparison[ii,:][3]
    
    zones = []
    for c in corclusters:
        zones.append('zone' + str(c))
    
    cvlim = np.max([abs(np.nanmin(minMap)),abs(np.nanmin(maxMap))])
    
    # Make plot
    fig, ax = plt.subplots(2,2, layout='constrained', figsize=figsize, sharex=True)

    # mean
    im0 = heatmap(meanMap, zones, zones, labelsize, ax=ax[0,0], cmap="coolwarm_r", vmin=-cvlim, vmax=cvlim, aspect=0.5)
    annotate_heatmap(im0, valfmt="{x:.10f}", size=textsize, threshold=20)
    ax[0,0].set_title('Mean pearson', size = titlesize,pad=30)
    
    # median
    im1 = heatmap(medianMap, zones, zones, labelsize, ax=ax[0,1], cmap="coolwarm_r", vmin=-cvlim, vmax=cvlim, aspect=0.5)
    annotate_heatmap(im1, valfmt="{x:.10f}", size=textsize, threshold=20)
    ax[0,1].set_title('Median pearson', size = titlesize,pad=30)
    
    # max
    im2 = heatmap(maxMap, zones, zones, labelsize, ax=ax[1,0], cmap="coolwarm_r", vmin=-cvlim, vmax=cvlim, aspect=0.5)
    annotate_heatmap(im2, valfmt="{x:.10f}", size=textsize, threshold=20)
    ax[1,0].set_title('Max pearson', size = titlesize,pad=30)
    
    # min
    im3 = heatmap(minMap, zones, zones, labelsize, ax=ax[1,1], cmap="coolwarm_r", vmin=-cvlim, vmax=cvlim, aspect=0.5)
    annotate_heatmap(im3, valfmt="{x:.10f}", size=textsize, threshold=20)
    ax[1,1].set_title('Min pearson', size = titlesize,pad=30)
    
    # Store
    plt.savefig(output_file, bbox_inches='tight', dpi=fig.dpi)
    plt.show()
    
#%% How long did the run take?
end = datetime.datetime.now()
td = round((end - start).total_seconds())
print("Script running time = " + str(timedelta(seconds=td)))
