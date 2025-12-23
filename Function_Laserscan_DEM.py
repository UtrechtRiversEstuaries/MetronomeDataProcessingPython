'''
Function to create NetCDF DEMs from the raw laserscan data, using geometry and linear algebra.

Python script written by Eise W. NOTA (finalized MAY 2025)

Calibration from CalibrateLaserCamera.py.

'''

# Importing packages
#import shutil
import os
import pandas as pd
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
import netCDF4 as nc
import warnings
from tqdm import tqdm
import datetime
from datetime import timedelta

# Set base directory of the network drive
pwd = r"\\..."

# Import functions
os.chdir(pwd + r"\\Laserscan_scripts\Functions_laserscan") # Setting the functions directory
from Function_undistort import undistort_laser_line

def Compute_DEM(inputFiles,writefolders,DEM_name,gridResols,yCoordinates,xCoordinates,percentiles,nrPoints,laserCamLens,
                laserPointerTr,laserCamTr,average,domingCorrection,domingFunction):
    '''
    This function produces a structure g (gridded bathymetry data) that contains grids with units in m.
    In order to prevent flooding of the drives, gridded data is limited to z values.
    x and y information is limited to ROI coordinates and cell size
    Any code processing this information should consider this and do additional computation of this information.
    
    Parameters
    ----------
    input_files : Source tif files of the raw laserscanner data
    writeFolders : Folders to store the NetCDF DEMs
    DEM_name  : String with output file name
    gridResols : grid resolutions in mm
    x & yCoordinates : information on the ROI we want to print
    nrPoints : whether we want to calculate the number of points, boolean
    laserCamLens : lens parameters of the laser camera
    laserPointerTr : estimated trajectory of the laser pointer positions and orientations
    laserCamTr : estimated trajectory of the laser camera positions and orientations
    average: instead of obtaining percentile elevation values, the average value is computed (percentiles set to [50])
    domingCorrection : do we apply doming correction
    domingFunction : if so, this is the elliptic paraboloid function which we use to apply the correction
    '''
    # Turn warnings off
    warnings.filterwarnings("ignore")
    
    #%% Load the input scan files into a single array
    # Loop through all the input files
    for i in range(len(inputFiles)): # Flip and rotate 90 degrees to make it match the desired coordinate system
        scanFile = np.flipud(np.rot90(cv2.imread(inputFiles[i],-1))) # -1 assures that imagery is read as 16-bit grayscale
        # We need image dimensions to construct an array that we want to fill with the raw scans
        if i == 0: # This is only required in the first timestep
            IMsize = scanFile.shape 
            scanDataRaw = np.empty([IMsize[0],IMsize[1]*len(inputFiles)])
        # Fill the array with the separate raw data files
        scanDataRaw[:,i*IMsize[1]:(i+1)*IMsize[1]] = scanFile
        del(scanFile)
    
    #%% To speed up the process, we're going to conduct the computations as much as possible within a single for loop through all columns
    # First define some constants 
    # What is the original image size? This is a given
    height, width = 3072, 4096
    # Define the x-range in pixels
    xRange = np.arange(0,width) # x in the image, so in the y-direction of metronome coordinates
    # What is the focal length of the laser camera?
    focalLength = laserCamLens.fy[0]
    
    # Start the loop # This can take a while so we'll show an alive progress bar to keep the spirits up
    # Set logging settings so there is not an overload of printed statements
    print('Computing the XYZ grid of ' + DEM_name)
    
    # Take track of how long it takes
    startGrid = datetime.datetime.now()
    # The XYZ_algebra function is provided below
    stackFlat = XYZ_algebra(laserCamTr,laserPointerTr,scanDataRaw,xRange,height,width,laserCamLens,focalLength,xCoordinates,yCoordinates)
    # How long did the algebra take?
    endGrid = datetime.datetime.now()
    tdGrid = round((endGrid - startGrid).total_seconds())
    print("XYZ grid computed in " + str(timedelta(seconds=tdGrid)))
    
    del laserCamTr, laserPointerTr, scanDataRaw, xRange, height, width, laserCamLens, focalLength
    
    #%% Now we have all our gridded data we can interpolate this data onto our desired grid
    # Now loop over the gridResols and writeFolders
    for h in range(len(gridResols)):
        # Only compute if the resolution does not already exist
        if not os.path.exists(writefolders[h] + '\\' + DEM_name):
            # Define the current writefolder and gridResol
            writefolder = writefolders[h]
            gridResol = gridResols[h]
            # Convert grid resolution from mm to m
            gridResol = gridResol/1000
            # Let the centres of each cell be in the middle
            desiredY = np.arange(yCoordinates[0]+gridResol/2,yCoordinates[1]+gridResol/2,gridResol)
            desiredX = np.arange(xCoordinates[0]+gridResol/2,xCoordinates[1]+gridResol/2,gridResol)
            # We want to store as little information as possible, so only beginning, end and step
            yInformation = np.array([yCoordinates[0]+gridResol/2,yCoordinates[1]+gridResol/2,gridResol])
            xInformation = np.array([xCoordinates[0]+gridResol/2,xCoordinates[1]+gridResol/2,gridResol])
            
            #%% Create the netCDF file
            netCDFFile = nc.Dataset(writefolder + '\\' + DEM_name,'w') 
            netCDFFile.title = DEM_name
            # Add axes attributes 
            netCDFFile.createDimension('X', len(xInformation))
            netCDFFile.createDimension('Y', len(yInformation))
            netCDFFile.createDimension('Xrange', len(desiredX))
            netCDFFile.createDimension('Yrange', len(desiredY))
            netCDFFile.createDimension('PcRange', len(percentiles))
            X_var = netCDFFile.createVariable('X-axis', np.float64, ('X'))
            X_var[:] = xInformation
            Y_var = netCDFFile.createVariable('Y-axis', np.float64, ('Y'))
            Y_var[:] = yInformation
            
            # Now define over the percentiles as variables
            actualPercentiles = []
            for pc in range(len(percentiles)):
                # What is the string of the current percentile?
                actualPercentiles.append("z" + str(percentiles[pc]))
        
            # Where are we going to store the Z_grids?
            Z_grids = np.full([len(desiredY),len(desiredX),len(percentiles)],np.nan)
            # Create the netCDF variable of Z and the percentiles
            Z_Var = netCDFFile.createVariable('Z-axis', np.float64, ('Yrange','Xrange','PcRange'))
            pcName_Var = netCDFFile.createVariable('Z percentiles', np.str_, ('PcRange'))
            pcName_Var[:] = np.array(actualPercentiles)
            
            # Do we want to store the number of points?
            if nrPoints == 1:
                nP_Var = netCDFFile.createVariable('Nr Points', np.int16, ('Yrange','Xrange'))
                nP_grid = np.zeros([len(desiredY),len(desiredX)]).astype(np.int16)
            
            #%% Now compute the desired grids
            print('Interpolating the elevation to grid size of ' + str(gridResol*1000) + ' mm')
            for i in tqdm (range(len(desiredX))):
                # First we have to find which x cells of the parent grid are represented in the current grid
                relevantXrange = [desiredX[i]-0.5*gridResol,desiredX[i]+0.5*gridResol]
                # Which X-cells does this correspond to? 
                xCells = np.where(np.logical_and(relevantXrange[0] <= stackFlat[0,:], stackFlat[0,:] < relevantXrange[1]))[0]             
                # Now loop over the Y-axis, but only if there are actual X columns available
                if len(xCells) != 0:
                    for j in range(len(desiredY)):
                        # First we have to find which x cells of the parent grid are represented in the current grid
                        relevantYrange = [desiredY[j]-0.5*gridResol,desiredY[j]+0.5*gridResol]
                        # The potential relevant X-columns are already determined, so we only use these values for determining the Y-rows
                        tempyGrid = stackFlat[1,:][xCells]
                        # Which of these cells are in the relevant Y-range? 
                        yCellsY = np.where(np.logical_and(relevantYrange[0] <= tempyGrid , tempyGrid < relevantYrange[1]))[0]
                        # Update the x and y cells for z with the minimum value
                        zCells = xCells[yCellsY]
                        # Take these Z values, but only if there are actually Y rows available
                        if len(zCells) != 0:
                            relevantZvalues = stackFlat[2,:][zCells]
                            # Now compute the relevant percentile grids
                            for pc in range(len(percentiles)):
                                # Do we want to make a grid based on average instead of median?
                                if average == 1:
                                    Z_grids[j,i,pc] = np.nanmean(relevantZvalues)
                                else:
                                    Z_grids[j,i,pc] = np.percentile(relevantZvalues, percentiles[pc])
                            # Do we want to store the nr of points?
                            if nrPoints == 1:
                                nP_grid[j,i] = len(relevantZvalues)
                            del(relevantZvalues)
                        else: #len(yCellsY) == 0
                            # Fill Nans
                            for pc in range(len(percentiles)):
                                Z_grids[j,i,pc] = np.nan
                            # Do we want to store the nr of points?
                            if nrPoints == 1:
                                nP_grid[j,i] = 0
                    del(relevantYrange,tempyGrid,yCellsY,zCells)
                else: #len(xCellsX) == 0
                    # Fill nans
                    for pc in range(len(percentiles)):
                        Z_grids[:,i,pc] = np.nan
                    # Do we want to store the nr of points?
                    if nrPoints == 1:
                        nP_grid[:,i] = 0
                del(relevantXrange,xCells) 
            
            # If we want to apply doming, set the settings
            # If we want to apply doming, set the settings
            if domingCorrection == 1:
                # Define the elliptic paraboloid function
                def domingFunc(x, y, a, b, c, d, e, f, offset): 
                    return a + b*x + c*y + d*x**2 + e*y**2 + f*x*y - offset
                # Define meshgrid
                x, y = np.meshgrid(desiredX, desiredY)
                # Define function variables
                a = domingFunction.a[0]; b = domingFunction.b[0]; c = domingFunction.c[0]; d = domingFunction.d[0]
                e = domingFunction.e[0]; f = domingFunction.f[0]; offset = domingFunction.offset[0]
                
            #%% Do some final interpolation at the NaNs
            Z_grids_interp = Z_grids
            for pc in range(len(percentiles)):
                # Make current grid 2D
                currentGrid = Z_grids[:,:,pc].reshape((Z_grids.shape[0],Z_grids.shape[1]))
                # For interpolation, we construct a DataFrame
                currentGridDf = pd.DataFrame(currentGrid)
                # We conduct linear interpolation along the x-axis, to prevent walls from affecting the interpolation. This should not be problematic as
                # the elevation data is already accounted for in along-x-axis variations of positions and orientations of the laser camera and pointer
                currentGridDf = currentGridDf.interpolate('linear',axis=1,limit_direction='both') # This fills all the NaNs, unless entire row is NaN (unlikely in our ROIs)
                # You can always test by a quick plot: plt.imshow(currentGridDf,interpolation='nearest')
                # Now transfer back to numpy array
                Z_grids_interp[:,:,pc] = np.array(currentGridDf)
                
                # Apply the doming correction, if set
                if domingCorrection == 1:
                    # Calculate Zcor
                    Zcor = domingFunc(x, y, a, b, c, d, e, f, offset)
                    # Apply to grid
                    Z_grids_interp[:,:,pc] = Z_grids_interp[:,:,pc] - Zcor
                
            del(Z_grids,currentGrid,currentGridDf) #ySliceRange,xSliceRange,arraySlice
            
            #%% Now write to the netCDF file
            # Write the Z variables
            Z_Var[:] = Z_grids_interp
            # Write the number of points (optionally)
            if nrPoints == 1:
                nP_Var[:] = nP_grid
            # Now we can close the netCDF file
            netCDFFile.close()
            
            print(DEM_name + ' at grid resolution ' + str(gridResol) + 'mm has been created and saved as netCDF')
        
        else:
            print(DEM_name + ' at grid resolution ' + str(gridResols[h]) + 'mm already exists')
    
    return print('all desired new DEMs have been created and saved as netCDF')
    

def XYZ_algebra(laserCamTr,laserPointerTr,scanDataRaw,xRange,height,width,laserCamLens,focalLength,xCoordinates,yCoordinates):
    '''
    Sub-function within the function above to compute the XYZ grid using computer graphics algebra.
    The equations are explained in SfM-photogrammetry/CG paper by Nota et al.
    '''
    #%% Before calculating with the raw data, we're going to define some timestep-dependent variables from our calibration data
    # First constuct a transformation matrix of the laser camera, based on its estimated trajectory
    # We need the positions and Euler angles provided in laserCamTr
    xCam = laserCamTr.X
    yCam = laserCamTr.Y
    zCam = laserCamTr.Z
    omegaCam = -np.radians(laserCamTr.omega)
    phiCam = -np.radians(laserCamTr.phi)
    kappaCam = -np.radians(laserCamTr.kappa)
    
    # First define the rotations around each axis separately
    # Note that we need to apply an y-axis rotation to our x-axis and vice versa
    rZCam = Rotation.from_euler('z', kappaCam).as_matrix()
    rXCam = Rotation.from_euler('y', phiCam).as_matrix()
    rYCam = Rotation.from_euler('x', omegaCam).as_matrix()
    # Multpily in the following (Agisoft) order to compute rotation matrices
    # Swap final two axes to obtain correct rotation matrices
    RCam =  np.swapaxes(rZCam @ rXCam @ rYCam,1,2)
    
    # Determine translation
    # Note that the focal point of our camera in the local coordinate system is (0,0,0)
    # So we subtract each translation component from zero
    pCam = 0 - (RCam[:,0,0]*xCam + RCam[:,1,0]*yCam + RCam[:,2,0]*zCam)
    qCam = 0 - (RCam[:,0,1]*xCam + RCam[:,1,1]*yCam + RCam[:,2,1]*zCam)
    rCam = 0 - (RCam[:,0,2]*xCam + RCam[:,1,2]*yCam + RCam[:,2,2]*zCam)
    translationCam = np.transpose(np.array([pCam,qCam,rCam]))
    
    # Construct transformation matrix and its inverse
    TCam = np.zeros([len(xCam),4,4])  # Create the empty 4x4 matrices
    TCam[:,0:3,0:3] = RCam            # Add Rotation matrices
    TCam[:,3,0:3] = translationCam    # Add translation
    TCam[:,3,3] = 1                   # Add the homgeneous component
    # Calculate the inverses
    invTCam = np.linalg.inv(TCam)
    
    # Also store all camera positions into a homogeneous coordinate system
    camCoords = np.swapaxes(np.transpose(np.array([xCam,yCam,zCam,np.ones(len(xCam))],ndmin=3)),1,2)

    del(omegaCam,phiCam,kappaCam,rZCam,rXCam,rYCam,RCam,pCam,qCam,rCam,laserCamTr,translationCam,TCam,xCam,yCam,zCam)
    
    #%% Now constuct a transformation matrix of the laser laser pointer, based on its estimated trajectory
    # We need the positions and Euler angles provided in laserPointerTr
    xPointer = laserPointerTr.X
    yPointer = laserPointerTr.Y
    zPointer = laserPointerTr.Z
    omegaPointer = -np.radians(laserPointerTr.omega)
    phiPointer = -np.radians(laserPointerTr.phi)
    kappaPointer = -np.radians(laserPointerTr.kappa)
    
    # First define the rotations around each axis separately
    # Note that we need to apply an y-axis rotation to our x-axis and vice versa
    rZPointer = Rotation.from_euler('z', kappaPointer).as_matrix()
    rXPointer = Rotation.from_euler('y', phiPointer).as_matrix()
    rYPointer = Rotation.from_euler('x',omegaPointer).as_matrix()
    # Multpily in the following (Agisoft) order to compute rotation matrices
    # Swap final two axes to obtain correct rotation matrices
    RPointer = np.swapaxes(rZPointer @ rXPointer @ rYPointer,1,2)
    
    # Determine translation
    # Note that the location  of our pointer in the local coordinate system is (0,0,0)
    # So we subtract each translation component from zero
    pPointer = 0 - (RPointer[:,0,0]*xPointer + RPointer[:,1,0]*yPointer + RPointer[:,2,0]*zPointer)
    qPointer = 0 - (RPointer[:,0,1]*xPointer + RPointer[:,1,1]*yPointer + RPointer[:,2,1]*zPointer)
    rPointer = 0 - (RPointer[:,0,2]*xPointer + RPointer[:,1,2]*yPointer + RPointer[:,2,2]*zPointer)
    translationPointer = np.transpose(np.array([pPointer,qPointer,rPointer]))
    
    # Construct transformation matrix and its inverse
    TPointer = np.zeros([len(xPointer),4,4])   # Create the empty 4x4 matrices
    TPointer[:,0:3,0:3] = RPointer             # Add Rotation matrices
    TPointer[:,3,0:3] = translationPointer     # Add translation
    TPointer[:,3,3] = 1                        # Add the homgeneous component
    # Calculate the inverses
    invTPointer = np.linalg.inv(TPointer)
    
    del(xPointer,yPointer,zPointer,omegaPointer,phiPointer,kappaPointer,rZPointer,laserPointerTr,
        rXPointer,rYPointer,RPointer,pPointer,qPointer,rPointer,translationPointer)
    
    #%% Now continue with the data and apply lens correction to the raw laser camera calibration data
    # Convert from 16-bit back to pixel data
    pixelData = 1699-(scanDataRaw/16)
    # 1699 is NaN values, so we ignore these
    pixelData[pixelData == 1699] = np.nan
    # Group xRange and pixelData into a coordinate system
    # First set the xRange to a grid
    xRangeGrid = np.transpose(np.tile(xRange, (len(pixelData[0,:]), 1)))
    # And stack
    pixelCoords = np.stack([xRangeGrid,pixelData])
    # Swap axes
    pixelCoords = np.swapaxes(pixelCoords, 0, 2)
    # Reshape to simple 3D structure
    pixelCoords = np.reshape(pixelCoords,(len(pixelCoords[0,:,:])*len(pixelCoords[:,0,:]),2))
    # Apply the lens correction
    lensCorrectedColumns = undistort_laser_line(pixelCoords, height, width, laserCamLens)
    
    del(pixelData,pixelCoords,xRangeGrid,laserCamLens,xRange)
    
    #%% This lens corrected data now tells where on the y-axis of the image, the laser camera sees the laser for each x-coordinate.
    # The coordinates here are in pixels.
    # To find the global X,Y and Z coordinates of this point, we need to do some coordinate transformations.
    # Let's first construct a unit vector in the local camera coordinate system for each pixel
    # Adjust the xpixel coordinates minus half the width to let focal point fit middle of the image
    xPixelCoords = lensCorrectedColumns[:,:,0] - width/2 
    # For the yPixelCoords, subtract from height/2 to let focal point fit middle of the image
    yPixelCoords = height/2 - lensCorrectedColumns[:,:,1]
    # Now construct the vector of the camera ray as if pixels are meters. We know that focal point is in coordinate [0,0,0]
    localPixelRayVectors = np.transpose(np.stack([xPixelCoords.ravel(),yPixelCoords.ravel(),np.full(np.shape(xPixelCoords),-focalLength).ravel()]))
    del(lensCorrectedColumns,xPixelCoords,yPixelCoords,focalLength,height,width)
    # Get the length of these vectors
    localPixelRayVectorLengths = np.sqrt(localPixelRayVectors[:,0]**2+localPixelRayVectors[:,1]**2+localPixelRayVectors[:,2]**2)
    # Transform vectors into unit vector with length 1
    localCamUnitCoords = localPixelRayVectors/localPixelRayVectorLengths[:,None]
    # Make coordinates homogeneous
    hLocalCamUnitCoords = np.ones((len(localCamUnitCoords[:,0]),4))
    hLocalCamUnitCoords[:,:3] = localCamUnitCoords
    # Reshape back into desired structure
    sLocalCamUnitCoords = np.reshape(hLocalCamUnitCoords,(len(scanDataRaw[0,:]),len(scanDataRaw[:,0]),4))
    # Get global coordinates of this unit point using the inverse of the camera transformation matrix
    globalUnitCoords = np.matmul(sLocalCamUnitCoords,invTCam)
    del(localPixelRayVectorLengths,localCamUnitCoords,hLocalCamUnitCoords,sLocalCamUnitCoords,invTCam,scanDataRaw)
    # Get local coordinates of this unit point in the laser pointer coordinate system using the laser pointer transformation matrix
    localPointerUnitCoords = np.matmul(globalUnitCoords,TPointer)
    # Also get local coordinates of the laser camera focal point in the laser pointer coordinate system using the laser pointer transformation matrix
    localPointerCamCoords = np.matmul(camCoords,TPointer)
    # Now determine the unit vector between camera and unit point in the local laser pointer coordinate system
    localPointerRayUnitVector = np.array([localPointerUnitCoords[:,:,0]-localPointerCamCoords[:,:,0],  # X
                                          localPointerUnitCoords[:,:,1]-localPointerCamCoords[:,:,1],  # Y
                                          localPointerUnitCoords[:,:,2]-localPointerCamCoords[:,:,2]]) # Z
    del(globalUnitCoords,camCoords,TPointer,localPointerUnitCoords)
    # We need to find the coordinates where this vector crosses Y = 0 in the coordinate system of the laser pointer
    # This is namely the plane across which the laser line is projected
    # This is a simple algebra solution. First find the timestep at which Y reaches 0
    tCamRayThroughLaserLine = (0-localPointerCamCoords[:,:,1])/localPointerRayUnitVector[1,:,:]
    # Apply this timestep to obtain local X and Z coordinates
    xBedPointerCoord = localPointerCamCoords[:,:,0]+localPointerRayUnitVector[0,:,:]*tCamRayThroughLaserLine
    zBedPointerCoord = localPointerCamCoords[:,:,2]+localPointerRayUnitVector[2,:,:]*tCamRayThroughLaserLine
    # Combine all coordinates into a homogeneous coordinate system
    bedCoords = np.stack((xBedPointerCoord,np.zeros(xBedPointerCoord.shape),zBedPointerCoord,np.ones(xBedPointerCoord.shape)),axis=2)    
    # Now use the inverse of the transformation matrix of the laser pointer to obtain the global coordinates of where the laser hits the bed
    globalInterceptCoords = np.matmul(bedCoords,invTPointer)
    # Now store the information in the grids
    xGrid = np.transpose(globalInterceptCoords[:,:,0])
    yGrid = np.transpose(globalInterceptCoords[:,:,1])
    zGrid = np.transpose(globalInterceptCoords[:,:,2])
    del(localPointerCamCoords,tCamRayThroughLaserLine,localPointerRayUnitVector,xBedPointerCoord,zBedPointerCoord,bedCoords,invTPointer,globalInterceptCoords)
    
    # Remove outliers
    xGrid[0 > zGrid] = np.nan   
    xGrid[0.42 < zGrid] = np.nan   
    yGrid[0 > zGrid] = np.nan   
    yGrid[0.42 < zGrid] = np.nan   
    zGrid[0 > zGrid] = np.nan   
    zGrid[0.42 < zGrid] = np.nan   
    
    # Flatten the relevant grids
    xFlat = np.ndarray.flatten(xGrid)
    yFlat = np.ndarray.flatten(yGrid)
    zFlat = np.ndarray.flatten(zGrid)
    stackFlat = np.stack((xFlat,yFlat,zFlat))
    del(xFlat,yFlat,zFlat,xGrid,yGrid,zGrid)
    
    # Remove NaNs
    stackFlat = stackFlat[:,~np.all(np.isnan(stackFlat), axis=0)]
    # Remove all points outside of the desired grid coordinates
    stackFlat = stackFlat[:,stackFlat[0,:]>=xCoordinates[0]] # x min
    stackFlat = stackFlat[:,stackFlat[0,:]<=xCoordinates[1]] # x max
    stackFlat = stackFlat[:,stackFlat[1,:]>=yCoordinates[0]] # y min
    stackFlat = stackFlat[:,stackFlat[1,:]<=yCoordinates[1]] # y max
    
    return stackFlat