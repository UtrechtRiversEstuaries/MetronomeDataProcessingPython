'''
Function to translate an Agisoft GeoTIFF to a NetCDF file with the same format as the LaserScan DEMs for the
Metronome in the Earth Simulation Laboratory

Python script written by Eise W. Nota (finalized SEPTEMBER 2024) 
'''

# Importing packages
import rasterio
import netCDF4 as nc
import numpy as np
import pandas as pd

def geoTIFF2NetCDF(writeDEMFolder, demTif_name, demNetCDF_name, gridRes, startXY, endXY):
    '''
    Parameters
    ----------
    writeDEMFolder      : Folder to save the DEMs
    demTif_name         : String with GeoTIFF DEM output file name
    demNetDF_name       : String with NetCDF DEM output file name
    gridRes             : Pixel/grid size in mm
    startXY             : Start coordinates of X and Y
    endXY               : End coordinates of X and Y
    '''
    
    #%% Load the baseModel-determined GeoTIFF into an array
    geo_tiff = rasterio.open(writeDEMFolder + demTif_name)
    geoTIFFArray = geo_tiff.read()[0]
    
    #%% Prepare the netCDF file information
    # Convert grid resolution from mm to m
    gridResol = gridRes/1000
    # Let the centres of each cell be in the middle
    desiredY = np.arange(startXY[1]+gridResol/2,endXY[1]+gridResol/2,gridResol)
    desiredX = np.arange(startXY[0]+gridResol/2,endXY[0]+gridResol/2,gridResol)
    # We want to store as little information as possible, so only beginning, end and step
    yInformation = np.array([startXY[1]+gridResol/2,endXY[1]+gridResol/2,gridResol])
    xInformation = np.array([startXY[0]+gridResol/2,endXY[0]+gridResol/2,gridResol])
    # Remove outliers
    #geoTIFFArray[0 > geoTIFFArray ] = np.nan   
    geoTIFFArray[0.42 < geoTIFFArray ] = np.nan 
    # Interpolate NaN values
    Z_grid_interp = geoTIFFArray
    # For interpolation, we construct a DataFrame
    currentGridDf = pd.DataFrame(Z_grid_interp)
    # We conduct linear interpolation along the x-axis, to prevent walls from affecting the interpolation. This should not be problematic as
    # the elevation data is already accounted for in along-x-axis variations of positions and orientations of the laser camera and pointer
    currentGridDf = currentGridDf.interpolate('linear',axis=1,limit_direction='both') # This fills all the NaNs, unless entire row is NaN (unlikely in our ROIs)
    # Also interpolate on the other axis, in case there is a full row of NaNs (likely bottom row)
    currentGridDf = currentGridDf.interpolate('linear',axis=0,limit_direction='both')
    # You can always test by a quick plot: plt.imshow(currentGridDf,interpolation='nearest')
    # Now transfer back to numpy array and flip it
    Z_grid_interp[:,:] = np.flipud(np.array(currentGridDf))
    
    #%% Create the netCDF file
    netCDFFile = nc.Dataset(writeDEMFolder + demNetCDF_name,'w') 
    netCDFFile.title = demNetCDF_name
    # Create dimensions
    netCDFFile.createDimension('X', len(yInformation))
    netCDFFile.createDimension('Y', len(xInformation))
    netCDFFile.createDimension('Xrange', len(desiredX))
    netCDFFile.createDimension('Yrange', len(desiredY))
    # Create and add variables
    X_var = netCDFFile.createVariable('X-axis', np.float64, ('X'))
    X_var[:] = xInformation
    Y_var = netCDFFile.createVariable('Y-axis', np.float64, ('Y'))
    Y_var[:] = yInformation
    Z_var = netCDFFile.createVariable('Z-axis', np.float64, ('Yrange','Xrange'))
    Z_var[:] = Z_grid_interp
    
    # Now we can close the netCDF file
    netCDFFile.close()
    
    return print('DEM has been saved as netCDF in ' + writeDEMFolder)