'''
Function to create NetCDF DEMs from the raw laserscan data, using geometry and linear algebra.

Python script written by Eise W. NOTA (finalized MARCH 2026)

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
import pyvista as pv
from scipy.spatial import KDTree

# Set base directory of the network drive
pwd = r"\\..."

# Import functions
os.chdir(pwd + r"\\Laserscan_scripts\Functions_laserscan") # Setting the functions directory
from Function_undistort import undistort_laser_line

def Compute_DEM(rootFolder,inputFiles,writefolders,DEM_name,gridResols,yCoordinates,xCoordinates,percentiles,nrPoints,
                laserCamLens,laserPointerTr,laserCamTr,average,domingCorrection,domingFunction,correctRawDataGlitch):
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
    correctRawDataGlitch : if so, delete incorrect (repetitive) columns from the raw data before data processing
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
    
    #%% Determine whether there are already known incorrect columns
    if os.path.exists(rootFolder + '\\rawDataGlitchCorrection.csv'):
        scanDataRaw = applyGlitchCorrection(scanDataRaw,rootFolder) # Function below
        
    # Else apply raw data glitch correction, if set
    elif correctRawDataGlitch == 1:
        scanDataRaw = correctGlitch(scanDataRaw,rootFolder) # Function below
    
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
    
    # Conduct interpolation # Functions below
    # Take track of how long it takes
    startInterp = datetime.datetime.now()
    print("Conducting interpolation through Nearest Neighbour method")      
    NNDinterp(stackFlat, gridResols, writefolders, DEM_name, xCoordinates, yCoordinates, percentiles, average, nrPoints, domingCorrection, domingFunction)
    
    #print("Conducting interpolation through Moving Window method")      
    #MWinterp(stackFlat, gridResols, writefolders, DEM_name, xCoordinates, yCoordinates, percentiles, average, nrPoints, domingCorrection, domingFunction)
    # Note: Moving Window is the old method and much slower than the Nearest Neighbour Method
    
    # How long did the interpolation take?
    endInterp = datetime.datetime.now()
    tdInterp = round((endInterp - startInterp).total_seconds())
    print("Interpolation conducted in " + str(timedelta(seconds=tdInterp)))
    
    return print('all desired new DEMs have been created and saved as netCDF')    

#%% Function of XYZ algebra
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

#%% Define the elliptic paraboloid function
def domingFunc(x, y, a, b, c, d, e, f, offset): 
    return a + b*x + c*y + d*x**2 + e*y**2 + f*x*y - offset

#%% Interpolate through Neirest Neighbour Interpolation
def NNDinterp(stackFlat, gridResols, writefolders, DEM_name, xCoordinates, yCoordinates, percentiles, average, nrPoints, domingCorrection, domingFunction):
    '''
    Sub-function within the function above to compute the XYZ grid using computer graphics algebra.
    The equations are explained in SfM-photogrammetry/CG paper by Nota et al.
    '''
    # Prepare pv grid of stackFlat XY (Z as 0)
    gridLaserXY = pv.StructuredGrid(stackFlat[0,:], stackFlat[1,:], np.zeros(np.size(stackFlat,1))) # XYZ
    
    # Now loop over the gridResols and writeFolders
    for h in range(len(gridResols)):
        # Define the current writefolder and gridResol
        writefolder = writefolders[h]
        gridResol = gridResols[h]
        # Convert grid resolution from mm to m
        gridResol = gridResol/1000
        # Only compute if the resolution does not already exist
        if not os.path.exists(writefolders[h] + '\\' + DEM_name):
            print('Interpolating the elevation to grid size of ' + str(gridResol*1000) + ' mm')
            # We want to store as little information as possible, so only beginning, end and step
            yInformation = np.array([yCoordinates[0]+gridResol/2,yCoordinates[1]+gridResol/2,gridResol])
            xInformation = np.array([xCoordinates[0]+gridResol/2,xCoordinates[1]+gridResol/2,gridResol])
            # Now make meshgrids for interpolation
            # Let the centres of each cell be in the middle
            desiredY = np.arange(yCoordinates[0]+gridResol/2,yCoordinates[1]+gridResol/2,gridResol)
            desiredX = np.arange(xCoordinates[0]+gridResol/2,xCoordinates[1]+gridResol/2,gridResol)
            # Make meshgrid
            xx, yy = np.meshgrid(desiredX,desiredY)
            # What are desired 1D indices for this meshgrid?
            dixd = np.arange(0,len(desiredX)*len(desiredY),1)
            # Make pv grid
            gridLasertTarget = pv.StructuredGrid(xx, yy, np.zeros(np.shape(xx)))
            
            # If we want to apply doming, set the settings
            if domingCorrection == 1:
                # Define meshgrid
                x, y = np.meshgrid(desiredX, desiredY)
                # Define function variables
                a = domingFunction.a[0]; b = domingFunction.b[0]; c = domingFunction.c[0]; d = domingFunction.d[0]
                e = domingFunction.e[0]; f = domingFunction.f[0]; offset = domingFunction.offset[0]
            
            # Calculate kd-tree
            tree = KDTree(gridLasertTarget.points)
            print("kd-tree constructed")
            # Query the kd-tree for nearest neighbors
            d_kdtree, idx = tree.query(gridLaserXY.points)
            print("Nearest neighbours identified")
            # Append idx to stackFlat
            z_array = np.vstack([idx,stackFlat])
            # Sort by idx 
            z_array = z_array[:,z_array[0,:].argsort()]
            # What are the unique indixes
            uid = np.unique(z_array[0,:],return_index=True)
            # Split Z values by unique indixes
            z_array = np.split(z_array[3,:],uid[1][1:])
            print("Cells sorted and split by grid resolution")
            
            # Do we want to store the nr of points?
            if nrPoints == 1:
                nrsPoints = np.array(list(map(lambda l: len(l), z_array)))
                # Redindex based on the desired indices
                nrsPoints = np.array(pd.DataFrame(nrsPoints,index=uid[0]).reindex(dixd))[:,-1]
                # Reshape to meshgrid
                nrsPoints = np.transpose(nrsPoints.reshape((xx.shape[1],xx.shape[0])))
                print("Nr of points computed")
            
            # Where are we going to store the Z_grids?
            Z_grids = np.full([len(desiredY),len(desiredX),len(percentiles)],np.nan)
            # Calculate relevant percentiles
            for pc in range(len(percentiles)):
                # Do we want to make a grid based on average instead of median?
                if average == 1:
                    z_array = np.array(list(map(lambda l: np.mean(l), z_array)))
                    print("Average values computed")
                else:
                    z_array = np.array(list(map(lambda l: np.nanpercentile(l, percentiles[pc]), z_array))) 
                    print("z" + str(percentiles[pc]) + " values computed")
                    
                # Redindex z based on the desired indices
                z_array = np.array(pd.DataFrame(z_array,index=uid[0]).reindex(dixd))[:,-1]
                # Reshape to meshgrid
                z_array = np.transpose(z_array.reshape((xx.shape[1],xx.shape[0])))
                # Conduct some interpolation, for which we construct a DataFrame
                z_array = pd.DataFrame(z_array)
                # We conduct linear interpolation along the x-axis, to prevent walls from affecting the interpolation. This should not be problematic as
                # the elevation data is already accounted for in along-x-axis variations of positions and orientations of the laser camera and pointer
                z_array = z_array.interpolate('linear',axis=1,limit_direction='both') # This fills all the NaNs, unless entire row is NaN (unlikely in our ROIs)
                # You can always test by a quick plot: plt.imshow(currentGridDf,interpolation='nearest')
                # Now transfer back to numpy array
                z_array = np.array(z_array)
                
                # Correct for doming, if set
                if domingCorrection == 1:
                    # Calculate Zcor
                    Zcor = domingFunc(x, y, a, b, c, d, e, f, offset)
                    # Apply to grid
                    z_array = z_array - Zcor
                
                # Append to Z-grids
                Z_grids[:,:,pc] = z_array
                print("Meshgrid computed for z" + str(percentiles[pc]))
                del(z_array)

            #% Now write all relevant information to NetCDF file
            netCDFFile = nc.Dataset(writefolder + '\\' + DEM_name,'w') 
            netCDFFile.title = DEM_name
            # Add axes attributes 
            netCDFFile.createDimension('X', len(xInformation))
            netCDFFile.createDimension('Y', len(yInformation))
            netCDFFile.createDimension('Xrange', len(desiredX))
            netCDFFile.createDimension('Yrange', len(desiredY))
            netCDFFile.createDimension('PcRange', len(percentiles))
            X_var = netCDFFile.createVariable('X-axis', np.float64, ('X'))
            Y_var = netCDFFile.createVariable('Y-axis', np.float64, ('Y'))
            Z_Var = netCDFFile.createVariable('Z-axis', np.float64, ('Yrange','Xrange','PcRange'))
            pcName_Var = netCDFFile.createVariable('Z percentiles', np.str_, ('PcRange'))
            # Do we want to store the number of points?
            if nrPoints == 1:
                nP_Var = netCDFFile.createVariable('Nr Points', np.int16, ('Yrange','Xrange'))

            # Now define over the percentiles as variables
            actualPercentiles = []
            for pc in range(len(percentiles)):
                # What is the string of the current percentile?
                actualPercentiles.append("z" + str(percentiles[pc]))
                
            # Write all information    
            X_var[:] = xInformation
            Y_var[:] = yInformation
            pcName_Var[:] = np.array(actualPercentiles)
            Z_Var[:] = Z_grids
            
            # Write the number of points (optionally)
            if nrPoints == 1:
                nP_Var[:] = nrsPoints
            # Now we can close the netCDF file
            netCDFFile.close()
            
            print('Interpolating complete of grid size of ' + str(gridResol*1000) + ' mm and stored as NetCDF')
        else:
            print('NetCDF file of grid size of ' + str(gridResol*1000) + ' mm already exists')
            
#%% Moving window interpolation function
def MWinterp(stackFlat, gridResols, writefolders, DEM_name, xCoordinates, yCoordinates, percentiles, average, nrPoints, domingCorrection, domingFunction):
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
                
            #% Do some final interpolation at the NaNs
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
            
            #% Now write to the netCDF file
            # Write the Z variables
            Z_Var[:] = Z_grids_interp
            # Write the number of points (optionally)
            if nrPoints == 1:
                nP_Var[:] = nP_grid
            # Now we can close the netCDF file
            netCDFFile.close()

#%% Function to correct for laserscan glitch
def correctGlitch(scanDataRaw,rootFolder):
    '''
    Correct for duplicate columns that are a glitch that occasionally occurs in the raw data collection
    '''
    # In case the glitch shows more unexpected behaviour
    panicButton = 0
    
    # Identify unique columns and their indices
    uniqueScan, indices, inverses, counts = np.unique(scanDataRaw, return_index=True, return_inverse=True, return_counts=True, axis=1)
    del(uniqueScan)
    
    # Sort indices
    indicesSorted = np.sort(indices)
    
    # Obtain indices of duplicate columns
    fullIndices = np.arange(0,np.shape(scanDataRaw)[1])
    duplicateIndices = np.delete(fullIndices, indicesSorted)
    
    # Are there no recurring counts?
    if len(np.where(counts>2)[0]) == 0:
        # As well as from the original columns
        originalCountLocs = np.where(counts>1)[0]
    else: # Loop over the amount of recurring cells
        originalCountLocsList = []
        for i in range(2,len(np.unique(counts))+1):
            # Loop again over the size of i
            for j in range(0,i-1):
                # Append to list
                originalCountLocsList.append(np.where(counts==i)[0])
        # Stack
        originalCountLocs = np.sort(np.hstack(originalCountLocsList))
        
    originalIndices = np.sort(indices[originalCountLocs])

    # Make a new scandata array of only zeros
    newScanArray = scanDataRaw.copy()
    # Set duplicate columns to zero
    newScanArray[:,duplicateIndices] = 0
    
    # The glitch doesn't start nor end at the boundaries of a column, so we need to identify where the problems occur in the neighbouring columns
    # Create a new duplicateIndices array with two additional indices 
    duplicateIndices2 = np.full(len(duplicateIndices)+2,-1)
    duplicateIndices2[1:-1] = duplicateIndices
    originalIndices2 = np.full(len(originalIndices)+2,-1)
    originalIndices2[1:-1] = originalIndices
    
    # Note: something very peculiar occurs that boundary columns can swap between successive duplicates and originals, so this needs to be identified first
    # So first store the information separately
    leftOriginals = []
    leftDuplicates = []
    rightOriginals = []
    rightDuplicates = []
    leftLocs = []
    rightLocs = []
    fullColumnSets = [[]]
    
    # Loop over the duplicateIndices2
    for i in range(1,len(duplicateIndices2)-1):
        # Determine if we're at a left boundary
        if abs(duplicateIndices2[i] - duplicateIndices2[i-1]) > 1:
            # Store leftColumns
            leftLocs.append(duplicateIndices2[i]-1)
            leftOriginals.append(newScanArray[:,originalIndices2[i]-1])
            leftDuplicates.append(newScanArray[:,duplicateIndices2[i]-1])
        
        # Append full column to current set
        fullColumnSets[-1].append(duplicateIndices2[i])
        
        # Determine if we're at a right boundary
        if abs(duplicateIndices2[i] - duplicateIndices2[i+1]) > 1:
            # Determine the leftColumns
            rightLocs.append(duplicateIndices2[i]+1)
            rightOriginals.append(newScanArray[:,originalIndices2[i]+1])
            rightDuplicates.append(newScanArray[:,duplicateIndices2[i]+1])
            
            # Create new column set
            fullColumnSets.append([])
    
    # Delete final column set
    del(fullColumnSets[-1])
        
    # Now determine which duplicate boundary to match with which original boundary
    leftIndices = []
    rightIndices = []
    # Loop over duplicates
    for i in range(len(leftDuplicates)):
        # Loop over left originals
        leftScores = []
        for j in range(len(leftOriginals)):
            # Determine the possible startRows per original
            equalLeftcells = np.where(leftOriginals[j]==leftDuplicates[i])[0]
            leftScores.append([len(equalLeftcells)])
            
            del equalLeftcells
            
        # Loop over right originals
        rightScores = []
        for j in range(len(leftOriginals)):
            # Determine the possible startRows per original
            equalRightcells = np.where(rightOriginals[j]==rightDuplicates[i])[0]
            rightScores.append([len(equalRightcells)])
            
            del equalRightcells
        
        # Change to array
        leftScores = np.array(leftScores)
        rightScores = np.array(rightScores)
        
        # Append appropriate index == highest similarity
        leftIndices.append(np.argmax(leftScores))
        rightIndices.append(np.argmax(rightScores))
    
    # If not all indices are unique, additional interference is required
    if len(np.unique(leftIndices)) != len(leftIndices) or len(np.unique(rightIndices)) != len(rightIndices):
        # If only one is shorter, we could copy it to the other one, if that matches the full length
        if len(np.unique(leftIndices)) == len(leftIndices):
            rightIndices = leftIndices.copy()
        elif len(np.unique(rightIndices)) == len(rightIndices):
            leftIndices = rightIndices.copy()
        else: # This specific DEM is too complicated 
            # Hit the panic button
            panicButton = 1
            #raise Exception('Left and right indices are incomplete, manually adjust in function code')
    
    # Where to store the final information
    storeInfo = [[]] # leftStartRow - duplicateColumns - rightEndRow - etc.
    
    # Now loop through the amount of incorrect sections
    for i in range(len(leftIndices)):
        # We first append the left boundary start row
        # If panicButton is set, we'll just kill the entire column    
        if panicButton == 1:
            leftStartRow = 0
        else:
            # Determine the decided equalLeftcell 
            equalLeftcells = np.where(leftOriginals[leftIndices[i]]==leftDuplicates[i])[0]
            # If all distances between values are equal to one, we'll just take the first relevant value
            if len(set(np.diff(equalLeftcells))) == 1:
                leftStartRow = equalLeftcells[0]
            else: # Look where the difference deviates
                leftStartRow = equalLeftcells[np.where(np.diff(equalLeftcells)>1)[0][-1]+1] # Add 1 as this is the cell from where we want to delete
            
        # Set all these values to zero in the leftDuplicate column in the newScanArray
        newScanArray[leftStartRow:,leftLocs[i]] = 0
        # Append start row to info
        storeInfo[-1].append(leftStartRow)
        
        # Now append the indices of incorrect columns
        for column in fullColumnSets[i]:
            storeInfo[-1].append(column)
            
        # Now append right boundary end row
        # If panicButton is set, we'll just kill the entire column    
        if panicButton == 1:
            rightEndRow = 4096
        else:
            # Determine the decided equalLeftcell 
            equalRightcells = np.where(rightOriginals[rightIndices[i]]==rightDuplicates[i])[0]
            # If all distances between values are equal to one, we'll just take the last relevant value
            if len(set(np.diff(equalRightcells))) == 1:
                rightEndRow = equalRightcells[-1]
            else: # Look where the difference deviates
                # Determine the first location (i.e. [1]) where the equalcells are continuous (i.e difference between subsequent cells == 1)
                rightEndRow = equalRightcells[np.where(np.diff(equalRightcells)>1)[0][0]]+1 # Add 1 as this is the cell excluded from deletion
            
        # Set all these values to zero in the rightDuplicate column in the newScanArray
        newScanArray[:rightEndRow,rightLocs[i]] = 0
        # Append end row to info
        storeInfo[-1].append(rightEndRow)
        
        # Append new row to storeInfo
        storeInfo.append([])
    
    # Delete final storeInfo
    del storeInfo[-1]
    
    #plt.imshow(newScanArray)
    
    # Write info to a csv file
    with open(rootFolder + '\\rawDataGlitchCorrection.csv', 'w', newline='') as storeFile:
        storeWriter = csv.writer(storeFile)
        # Store info per section as row
        for infoRow in range(len(storeInfo)):
            storeWriter.writerow(map(lambda x: [x], storeInfo[infoRow]))    
    
    # Now test whether file is empty
    try: 
        pd.read_csv(rootFolder + '\\rawDataGlitchCorrection.csv',sep='\t',header=None)
        print('Raw scan data glitch columns loaded and corrected')
    except pd.errors.EmptyDataError:
        print('There are no Raw scan data glitch columns identified')
    
    return newScanArray

#%% Function to apply for the glitch correction
def applyGlitchCorrection(scanDataRaw,rootFolder):
    '''
    Load already existing information about the glitched columns and correct for them
    '''
    # First test whether file is empty
    try: 
        correctionInfo = pd.read_csv(rootFolder + '\\rawDataGlitchCorrection.csv',sep='\t',header=None)
    
        # Loop through the rows
        for r in range(correctionInfo.shape[0]):
            # Take current section and split at delimeters ','
            currentSection = correctionInfo.iloc[r].str.split(pat=',',expand=True)
            # Set current column to integers
            currentSection = currentSection.iloc[0,:].str.strip('[]').astype(int)
            # Take left startRow
            leftStartRow = currentSection.iloc[0]
            # Set all these values to zero in the scanDataRaw
            scanDataRaw[leftStartRow:,currentSection.iloc[1]-1] = 0
            # Take the right endRow
            rightEndRow = currentSection.iloc[-1]
            # Set all these values to zero in the scanDataRaw
            scanDataRaw[:rightEndRow,currentSection.iloc[-2]+1] = 0
            # Set all other columns to zero
            scanDataRaw[:,currentSection.iloc[1]:currentSection.iloc[-2]+1] = 0
            print('Raw scan data glitch columns loaded and corrected')
    
    except pd.errors.EmptyDataError:
        print('There are no Raw scan data glitch columns identified')
    
    #plt.imshow(scanDataRaw)
    return scanDataRaw
